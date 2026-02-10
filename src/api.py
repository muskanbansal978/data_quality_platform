"""
REST API for the Data Quality Platform.

Run with:
    python -m src.api
    uvicorn src.api:app --reload

Endpoints:
    GET  /api/health                  Health check
    POST /api/pipeline/run            Trigger pipeline run (async)
    GET  /api/pipeline/status/{id}    Poll pipeline job status
    GET  /api/anomalies               List anomalies (with filters)
    GET  /api/anomalies/{id}          Get single anomaly with explanation
    GET  /api/profiles                List available profiles
    GET  /api/profiles/{filename}     Get specific profile
    GET  /api/data/files              List data files
    GET  /api/config/alerts           Read alert config
    PUT  /api/config/alerts           Update alert config
    GET  /api/config/llm              Read LLM config
    PUT  /api/config/llm              Update LLM config
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.alerting import load_alert_config, save_alert_config, send_alerts
from src.anomaly_detector import (
    DEFAULT_MIN_HISTORY_DAYS,
    compute_daily_profiles,
    detect_all_anomalies,
    detect_date_column,
)
from src.data_loader import get_file_info, UniversalDataLoader
from src.llm_explainer import explain_anomaly, explain_anomaly_mock, get_provider_and_model

app = FastAPI(title="Data Quality Platform API", version="1.0.0")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROFILES_DIR = PROJECT_ROOT / "profiles"
LLM_CONFIG_FILE = PROJECT_ROOT / "config" / "llm_config.json"

# --- In-memory stores ---
pipeline_jobs: Dict[str, Dict[str, Any]] = {}
anomaly_index: Dict[str, Dict[str, Any]] = {}


# --- Pydantic models ---

class PipelineRunRequest(BaseModel):
    generate_data: bool = False


class AlertConfigUpdate(BaseModel):
    config: Dict[str, Any]


class LLMConfigUpdate(BaseModel):
    config: Dict[str, Any]


# --- Health ---

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# --- Pipeline ---

def _run_pipeline_job(job_id: str, generate: bool):
    """Execute pipeline in background. Updates pipeline_jobs dict."""
    from src.data_generator import main as generate_data
    from src.data_profiler import main as profile_data

    pipeline_jobs[job_id]["status"] = "running"
    pipeline_jobs[job_id]["started_at"] = datetime.now().isoformat()

    try:
        if generate:
            generate_data()
        profile_data()

        # Run anomaly detection on all data files
        all_anomalies: Dict[str, list] = {}
        all_explained: List[tuple] = []

        csv_files = sorted(DATA_DIR.glob("*.csv"))
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            date_col = detect_date_column(df)
            if date_col is None:
                continue

            profiles = compute_daily_profiles(df, date_col)
            if len(profiles) < DEFAULT_MIN_HISTORY_DAYS:
                continue

            # Auto-detect ID column
            id_column = None
            for col in df.columns:
                if col.endswith("_id") and pd.api.types.is_numeric_dtype(df[col]):
                    id_column = col
                    break

            anomalies = detect_all_anomalies(
                profiles, df=df, date_column=date_col,
                check_freshness=True, check_correlations=True,
                check_sequences=id_column is not None, id_column=id_column,
            )

            if anomalies:
                all_anomalies[csv_file.name] = anomalies

                # Explain anomalies
                try:
                    provider, model, use_mock, api_key, custom_config = get_provider_and_model()
                except (FileNotFoundError, ValueError):
                    use_mock = True
                    provider = model = api_key = custom_config = None

                for anomaly in anomalies:
                    try:
                        if use_mock:
                            explanation = explain_anomaly_mock(anomaly)
                        else:
                            explanation = explain_anomaly(
                                anomaly, provider_name=provider, model=model,
                                api_key=api_key, custom_config=custom_config,
                            )
                    except Exception:
                        explanation = explain_anomaly_mock(anomaly)
                    all_explained.append((anomaly, explanation))

        # Populate anomaly index
        anomaly_index.clear()
        for file_name, anomalies_list in all_anomalies.items():
            for anomaly in anomalies_list:
                aid = str(uuid.uuid4())
                # Find matching explanation
                explanation = None
                for a, e in all_explained:
                    if a is anomaly:
                        explanation = e
                        break
                anomaly_index[aid] = {
                    "id": aid,
                    "file": file_name,
                    "type": anomaly.get("type"),
                    "severity": anomaly.get("severity"),
                    "date": str(anomaly.get("date", "")),
                    "column": anomaly.get("column"),
                    "message": anomaly.get("message", ""),
                    "expected": anomaly.get("expected"),
                    "actual": anomaly.get("actual"),
                    "z_score": anomaly.get("z_score"),
                    "explanation": explanation,
                }

        # Send alerts
        flat_anomalies = [a for lst in all_anomalies.values() for a in lst]
        alert_results = {}
        if flat_anomalies:
            alert_results = send_alerts(flat_anomalies, all_explained)

        pipeline_jobs[job_id]["status"] = "completed"
        pipeline_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        pipeline_jobs[job_id]["anomaly_count"] = len(anomaly_index)
        pipeline_jobs[job_id]["alert_results"] = alert_results

    except Exception as e:
        pipeline_jobs[job_id]["status"] = "failed"
        pipeline_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        pipeline_jobs[job_id]["error"] = str(e)


@app.post("/api/pipeline/run")
async def trigger_pipeline(request: PipelineRunRequest, background_tasks: BackgroundTasks):
    """Start a pipeline run in the background. Returns a job ID for polling."""
    # Reject if a pipeline is already running
    running = any(j["status"] == "running" for j in pipeline_jobs.values())
    if running:
        raise HTTPException(status_code=409, detail="A pipeline run is already in progress")

    job_id = str(uuid.uuid4())
    pipeline_jobs[job_id] = {
        "status": "pending",
        "started_at": None,
        "completed_at": None,
        "error": None,
        "anomaly_count": None,
        "alert_results": None,
    }
    background_tasks.add_task(_run_pipeline_job, job_id, request.generate_data)
    return {"job_id": job_id, "status": "pending"}


@app.get("/api/pipeline/status/{job_id}")
async def get_pipeline_status(job_id: str):
    """Poll the status of a pipeline run."""
    if job_id not in pipeline_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **pipeline_jobs[job_id]}


# --- Anomalies ---

@app.get("/api/anomalies")
async def list_anomalies(
    severity: Optional[str] = Query(None, description="Filter by severity (low, medium, high, critical)"),
    type: Optional[str] = Query(None, description="Filter by anomaly type"),
    date_from: Optional[str] = Query(None, description="Filter from date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Filter to date (YYYY-MM-DD)"),
    file: Optional[str] = Query(None, description="Filter by source file name"),
):
    """List detected anomalies with optional filters."""
    results = list(anomaly_index.values())

    if severity:
        results = [a for a in results if a["severity"] == severity]
    if type:
        results = [a for a in results if a["type"] == type]
    if file:
        results = [a for a in results if a["file"] == file]
    if date_from:
        results = [a for a in results if a["date"] >= date_from]
    if date_to:
        results = [a for a in results if a["date"] <= date_to]

    return {"total": len(results), "anomalies": results}


@app.get("/api/anomalies/{anomaly_id}")
async def get_anomaly(anomaly_id: str):
    """Get a single anomaly with its explanation."""
    if anomaly_id not in anomaly_index:
        raise HTTPException(status_code=404, detail="Anomaly not found")
    return anomaly_index[anomaly_id]


# --- Profiles ---

@app.get("/api/profiles")
async def list_profiles():
    """List available profile files."""
    if not PROFILES_DIR.exists():
        return {"profiles": []}

    profiles = []
    for f in sorted(PROFILES_DIR.glob("*.json")):
        try:
            with open(f) as fp:
                data = json.load(fp)
            profiles.append({
                "filename": f.name,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 4),
                "row_count": data.get("row_count"),
                "column_count": data.get("column_count"),
            })
        except (json.JSONDecodeError, OSError):
            profiles.append({"filename": f.name, "error": "Could not read profile"})

    return {"profiles": profiles}


@app.get("/api/profiles/{filename}")
async def get_profile(filename: str):
    """Get a specific profile by filename."""
    profile_path = PROFILES_DIR / filename
    if not profile_path.exists() or not profile_path.suffix == ".json":
        raise HTTPException(status_code=404, detail="Profile not found")

    # Prevent path traversal
    if ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    try:
        with open(profile_path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Could not parse profile JSON")


# --- Data files ---

@app.get("/api/data/files")
async def list_data_files():
    """List available data files with metadata."""
    if not DATA_DIR.exists():
        return {"files": []}

    files = []
    for f in sorted(DATA_DIR.iterdir()):
        if f.is_file():
            files.append(get_file_info(f))

    return {"files": files}


# --- Config: Alerts ---

@app.get("/api/config/alerts")
async def get_alert_config():
    """Read alert configuration."""
    try:
        config = load_alert_config()
        return {"config": config}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Alert config not found")


@app.put("/api/config/alerts")
async def update_alert_config(request: AlertConfigUpdate):
    """Update alert configuration."""
    save_alert_config(request.config)
    return {"status": "updated", "config": request.config}


# --- Config: LLM ---

@app.get("/api/config/llm")
async def get_llm_config():
    """Read LLM configuration."""
    if not LLM_CONFIG_FILE.exists():
        raise HTTPException(status_code=404, detail="LLM config not found")
    with open(LLM_CONFIG_FILE) as f:
        config = json.load(f)
    # Mask API keys in response
    for key, val in config.items():
        if isinstance(val, dict) and "api_key" in val and val["api_key"]:
            val["api_key"] = "***" + val["api_key"][-4:]
    return {"config": config}


@app.put("/api/config/llm")
async def update_llm_config(request: LLMConfigUpdate):
    """Update LLM configuration."""
    with open(LLM_CONFIG_FILE, "w") as f:
        json.dump(request.config, f, indent=2)
    return {"status": "updated"}


# --- Entry point ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
