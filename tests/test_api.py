"""Tests for the REST API."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api import app, anomaly_index, pipeline_jobs

client = TestClient(app)


# --- Fixtures ---

@pytest.fixture(autouse=True)
def clear_state():
    """Clear in-memory stores before each test."""
    pipeline_jobs.clear()
    anomaly_index.clear()
    yield
    pipeline_jobs.clear()
    anomaly_index.clear()


@pytest.fixture
def populated_anomalies():
    """Populate the anomaly index with sample data."""
    anomaly_index["a1"] = {
        "id": "a1", "file": "orders.csv", "type": "volume_drop", "severity": "critical",
        "date": "2026-02-07", "column": None, "message": "Row count dropped by 85%",
        "expected": 10000, "actual": 1500, "z_score": 4.2,
        "explanation": {"root_cause": "ETL job failed", "confidence": "high",
                        "suggested_actions": ["Check ETL logs"], "business_impact": "HIGH"},
    }
    anomaly_index["a2"] = {
        "id": "a2", "file": "orders.csv", "type": "null_rate_increase", "severity": "high",
        "date": "2026-02-07", "column": "customer_id", "message": "Null rate increased to 20%",
        "expected": 0.01, "actual": 0.20, "z_score": 3.5, "explanation": None,
    }
    anomaly_index["a3"] = {
        "id": "a3", "file": "products.csv", "type": "mean_shift", "severity": "medium",
        "date": "2026-02-06", "column": "price", "message": "Mean shifted",
        "expected": 50.0, "actual": 75.0, "z_score": 2.8, "explanation": None,
    }


# --- Health ---

class TestHealth:
    def test_health_check(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


# --- Pipeline ---

class TestPipeline:
    def test_trigger_returns_job_id(self):
        with patch("src.api._run_pipeline_job"):
            resp = client.post("/api/pipeline/run", json={"generate_data": False})
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_rejects_concurrent_runs(self):
        pipeline_jobs["existing"] = {"status": "running"}
        resp = client.post("/api/pipeline/run", json={"generate_data": False})
        assert resp.status_code == 409

    def test_status_not_found(self):
        resp = client.get("/api/pipeline/status/nonexistent")
        assert resp.status_code == 404

    def test_status_found(self):
        pipeline_jobs["test-job"] = {
            "status": "completed", "started_at": "2026-02-08T10:00:00",
            "completed_at": "2026-02-08T10:01:00", "error": None,
            "anomaly_count": 5, "alert_results": {"slack": True},
        }
        resp = client.get("/api/pipeline/status/test-job")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["anomaly_count"] == 5


# --- Anomalies ---

class TestAnomalies:
    def test_list_empty(self):
        resp = client.get("/api/anomalies")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_list_all(self, populated_anomalies):
        resp = client.get("/api/anomalies")
        assert resp.json()["total"] == 3

    def test_filter_by_severity(self, populated_anomalies):
        resp = client.get("/api/anomalies?severity=critical")
        data = resp.json()
        assert data["total"] == 1
        assert data["anomalies"][0]["type"] == "volume_drop"

    def test_filter_by_type(self, populated_anomalies):
        resp = client.get("/api/anomalies?type=mean_shift")
        assert resp.json()["total"] == 1

    def test_filter_by_file(self, populated_anomalies):
        resp = client.get("/api/anomalies?file=products.csv")
        assert resp.json()["total"] == 1

    def test_filter_by_date_range(self, populated_anomalies):
        resp = client.get("/api/anomalies?date_from=2026-02-07")
        assert resp.json()["total"] == 2

    def test_get_single(self, populated_anomalies):
        resp = client.get("/api/anomalies/a1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "volume_drop"
        assert data["explanation"]["root_cause"] == "ETL job failed"

    def test_get_not_found(self):
        resp = client.get("/api/anomalies/nonexistent")
        assert resp.status_code == 404


# --- Profiles ---

class TestProfiles:
    def test_list_profiles(self):
        resp = client.get("/api/profiles")
        assert resp.status_code == 200
        assert "profiles" in resp.json()

    def test_get_profile_not_found(self):
        resp = client.get("/api/profiles/nonexistent.json")
        assert resp.status_code == 404

    def test_path_traversal_rejected(self):
        resp = client.get("/api/profiles/..%2F..%2Fetc%2Fpasswd")
        assert resp.status_code in (400, 404)  # blocked either way


# --- Data files ---

class TestDataFiles:
    def test_list_data_files(self):
        resp = client.get("/api/data/files")
        assert resp.status_code == 200
        assert "files" in resp.json()


# --- Config: Alerts ---

class TestAlertConfig:
    def test_get_alert_config(self):
        resp = client.get("/api/config/alerts")
        assert resp.status_code == 200
        assert "config" in resp.json()

    def test_update_alert_config(self):
        new_config = {
            "enabled": True,
            "severity_threshold": "high",
            "anomaly_types": [],
            "channels": {"slack": {"enabled": False}, "email": {"enabled": False}, "webhook": {"enabled": False}},
        }
        resp = client.put("/api/config/alerts", json={"config": new_config})
        assert resp.status_code == 200
        assert resp.json()["status"] == "updated"

        # Verify it was saved
        resp2 = client.get("/api/config/alerts")
        assert resp2.json()["config"]["severity_threshold"] == "high"

        # Restore original
        original = {
            "enabled": True, "severity_threshold": "medium", "anomaly_types": [],
            "channels": {
                "slack": {"enabled": False, "webhook_url": "", "channel": "#data-quality-alerts", "mention_on_critical": True},
                "email": {"enabled": False, "smtp_host": "smtp.gmail.com", "smtp_port": 587, "smtp_user": "", "smtp_password": "", "use_tls": True, "from_address": "", "to_addresses": []},
                "webhook": {"enabled": False, "url": "", "method": "POST", "headers": {}, "include_explanations": True},
            },
        }
        client.put("/api/config/alerts", json={"config": original})


# --- Config: LLM ---

class TestLLMConfig:
    def test_get_llm_config(self):
        resp = client.get("/api/config/llm")
        assert resp.status_code == 200
        assert "config" in resp.json()
