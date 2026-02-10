"""Tests for the alerting module."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.alerting import (
    SEVERITY_ORDER,
    build_alert_payload,
    filter_anomalies,
    load_alert_config,
    save_alert_config,
    send_alerts,
    send_email_alert,
    send_slack_alert,
    send_webhook_alert,
)


# --- Fixtures ---

@pytest.fixture
def sample_anomalies():
    return [
        {"type": "volume_drop", "severity": "critical", "date": datetime(2026, 2, 7), "column": None, "message": "Row count dropped by 85%", "expected": 10000, "actual": 1500, "z_score": 4.2},
        {"type": "null_rate_increase", "severity": "high", "date": datetime(2026, 2, 7), "column": "customer_id", "message": "Null rate increased to 20%", "expected": 0.01, "actual": 0.20, "z_score": 3.5},
        {"type": "mean_shift", "severity": "medium", "date": datetime(2026, 2, 6), "column": "price", "message": "Mean shifted from 50 to 75", "expected": 50.0, "actual": 75.0, "z_score": 2.8},
        {"type": "string_length_anomaly", "severity": "low", "date": datetime(2026, 2, 6), "column": "name", "message": "Avg string length changed", "expected": 10.0, "actual": 15.0, "z_score": 1.5},
    ]


@pytest.fixture
def sample_explanations(sample_anomalies):
    return [
        (sample_anomalies[0], {"root_cause": "ETL job failed", "confidence": "high", "suggested_actions": ["Check ETL logs"], "business_impact": "Revenue reporting incomplete"}),
        (sample_anomalies[1], {"root_cause": "Schema migration issue", "confidence": "medium", "suggested_actions": ["Verify migration"], "business_impact": "Customer data gaps"}),
    ]


@pytest.fixture
def alert_config():
    return {
        "enabled": True,
        "severity_threshold": "medium",
        "anomaly_types": [],
        "channels": {
            "slack": {"enabled": True, "webhook_url": "https://hooks.slack.com/test", "channel": "#test", "mention_on_critical": True},
            "email": {"enabled": False},
            "webhook": {"enabled": True, "url": "https://example.com/webhook", "method": "POST", "headers": {}, "include_explanations": True},
        },
    }


# --- filter_anomalies ---

class TestFilterAnomalies:
    def test_filter_by_severity_medium(self, sample_anomalies):
        result = filter_anomalies(sample_anomalies, severity_threshold="medium")
        assert len(result) == 3
        assert all(SEVERITY_ORDER[a["severity"]] >= SEVERITY_ORDER["medium"] for a in result)

    def test_filter_by_severity_high(self, sample_anomalies):
        result = filter_anomalies(sample_anomalies, severity_threshold="high")
        assert len(result) == 2

    def test_filter_by_severity_critical(self, sample_anomalies):
        result = filter_anomalies(sample_anomalies, severity_threshold="critical")
        assert len(result) == 1
        assert result[0]["type"] == "volume_drop"

    def test_filter_by_severity_low_includes_all(self, sample_anomalies):
        result = filter_anomalies(sample_anomalies, severity_threshold="low")
        assert len(result) == 4

    def test_filter_by_type(self, sample_anomalies):
        result = filter_anomalies(sample_anomalies, severity_threshold="low", anomaly_types=["volume_drop", "mean_shift"])
        assert len(result) == 2
        assert {a["type"] for a in result} == {"volume_drop", "mean_shift"}

    def test_filter_empty_types_includes_all(self, sample_anomalies):
        result = filter_anomalies(sample_anomalies, severity_threshold="low", anomaly_types=[])
        assert len(result) == 4

    def test_filter_no_matches(self, sample_anomalies):
        result = filter_anomalies(sample_anomalies, severity_threshold="critical", anomaly_types=["mean_shift"])
        assert len(result) == 0


# --- build_alert_payload ---

class TestBuildAlertPayload:
    def test_payload_structure(self, sample_anomalies):
        payload = build_alert_payload(sample_anomalies)
        assert "timestamp" in payload
        assert "summary" in payload
        assert payload["anomaly_count"] == 4
        assert "severity_breakdown" in payload
        assert len(payload["anomalies"]) == 4

    def test_severity_breakdown(self, sample_anomalies):
        payload = build_alert_payload(sample_anomalies)
        assert payload["severity_breakdown"] == {"critical": 1, "high": 1, "medium": 1, "low": 1}

    def test_summary_format(self, sample_anomalies):
        payload = build_alert_payload(sample_anomalies)
        assert "4 anomalies detected" in payload["summary"]
        assert "1 critical" in payload["summary"]

    def test_with_explanations(self, sample_anomalies, sample_explanations):
        payload = build_alert_payload(sample_anomalies, sample_explanations)
        # First anomaly should have root_cause
        assert payload["anomalies"][0].get("root_cause") == "ETL job failed"
        # Third anomaly should not have root_cause
        assert payload["anomalies"][2].get("root_cause") is None

    def test_empty_anomalies(self):
        payload = build_alert_payload([])
        assert payload["anomaly_count"] == 0
        assert payload["anomalies"] == []


# --- send_slack_alert ---

class TestSendSlackAlert:
    @patch("src.alerting.urllib.request.urlopen")
    def test_success(self, mock_urlopen, sample_anomalies):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        payload = build_alert_payload(sample_anomalies)
        config = {"webhook_url": "https://hooks.slack.com/test", "channel": "#test", "mention_on_critical": True}
        assert send_slack_alert(payload, config) is True

    def test_no_webhook_url(self, sample_anomalies):
        payload = build_alert_payload(sample_anomalies)
        assert send_slack_alert(payload, {"webhook_url": ""}) is False

    @patch("src.alerting.urllib.request.urlopen", side_effect=Exception("Connection refused"))
    def test_failure(self, mock_urlopen, sample_anomalies):
        payload = build_alert_payload(sample_anomalies)
        config = {"webhook_url": "https://hooks.slack.com/test"}
        assert send_slack_alert(payload, config) is False


# --- send_webhook_alert ---

class TestSendWebhookAlert:
    @patch("src.alerting.urllib.request.urlopen")
    def test_success(self, mock_urlopen, sample_anomalies):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        payload = build_alert_payload(sample_anomalies)
        config = {"url": "https://example.com/hook", "method": "POST", "headers": {}, "include_explanations": True}
        assert send_webhook_alert(payload, config) is True

    def test_no_url(self, sample_anomalies):
        payload = build_alert_payload(sample_anomalies)
        assert send_webhook_alert(payload, {"url": ""}) is False

    @patch("src.alerting.urllib.request.urlopen")
    def test_strips_explanations_when_configured(self, mock_urlopen, sample_anomalies, sample_explanations):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        payload = build_alert_payload(sample_anomalies, sample_explanations)
        config = {"url": "https://example.com/hook", "include_explanations": False}
        send_webhook_alert(payload, config)

        # Verify the sent data doesn't contain explanation fields
        call_args = mock_urlopen.call_args
        sent_data = json.loads(call_args[0][0].data)
        for a in sent_data["anomalies"]:
            assert "root_cause" not in a


# --- send_alerts (orchestrator) ---

class TestSendAlerts:
    def test_disabled_config(self, sample_anomalies):
        config = {"enabled": False, "channels": {}}
        result = send_alerts(sample_anomalies, config=config)
        assert result == {}

    def test_no_matching_anomalies(self):
        anomalies = [{"type": "volume_drop", "severity": "low", "date": datetime(2026, 2, 7), "message": "test"}]
        config = {"enabled": True, "severity_threshold": "critical", "anomaly_types": [], "channels": {}}
        result = send_alerts(anomalies, config=config)
        assert result == {}

    @patch("src.alerting.send_slack_alert", return_value=True)
    @patch("src.alerting.send_webhook_alert", return_value=True)
    def test_sends_to_enabled_channels(self, mock_webhook, mock_slack, sample_anomalies, alert_config):
        result = send_alerts(sample_anomalies, config=alert_config)
        assert result == {"slack": True, "webhook": True}
        assert "email" not in result  # email is disabled

    @patch("src.alerting.send_slack_alert", return_value=False)
    def test_reports_failure(self, mock_slack, sample_anomalies):
        config = {
            "enabled": True, "severity_threshold": "low", "anomaly_types": [],
            "channels": {"slack": {"enabled": True, "webhook_url": "https://test"}, "email": {"enabled": False}, "webhook": {"enabled": False}},
        }
        result = send_alerts(sample_anomalies, config=config)
        assert result == {"slack": False}
