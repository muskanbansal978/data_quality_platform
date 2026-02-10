"""
Alerting system for data quality anomalies.

Supports Slack (incoming webhook), Email (SMTP), and generic Webhook channels.
Configuration: config/alert_config.json

Usage:
    from src.alerting import send_alerts
    results = send_alerts(anomalies, explanations)
    # results: {"slack": True, "email": False, "webhook": True}
"""

import json
import smtplib
import ssl
import urllib.request
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Optional

ALERT_CONFIG_FILE = Path(__file__).resolve().parents[1] / "config" / "alert_config.json"

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def load_alert_config() -> dict[str, Any]:
    """Load alert configuration from JSON file."""
    if not ALERT_CONFIG_FILE.exists():
        raise FileNotFoundError(f"Alert config not found: {ALERT_CONFIG_FILE}")
    with open(ALERT_CONFIG_FILE) as f:
        return json.load(f)


def save_alert_config(config: dict[str, Any]) -> None:
    """Save alert configuration to JSON file."""
    with open(ALERT_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def filter_anomalies(
    anomalies: list[dict[str, Any]],
    severity_threshold: str = "medium",
    anomaly_types: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Filter anomalies by severity threshold and optional type whitelist.

    Args:
        anomalies: List of anomaly dicts (must have 'severity' and 'type' keys).
        severity_threshold: Minimum severity to include ('low', 'medium', 'high', 'critical').
        anomaly_types: If non-empty, only include these anomaly types.

    Returns:
        Filtered list of anomalies.
    """
    threshold = SEVERITY_ORDER.get(severity_threshold, 1)
    filtered = [
        a for a in anomalies
        if SEVERITY_ORDER.get(a.get("severity", "low"), 0) >= threshold
    ]
    if anomaly_types:
        filtered = [a for a in filtered if a.get("type") in anomaly_types]
    return filtered


def build_alert_payload(
    anomalies: list[dict[str, Any]],
    explanations: Optional[list[tuple[dict, dict]]] = None,
) -> dict[str, Any]:
    """Build a structured alert payload from anomalies and optional explanations.

    Returns:
        Dict with summary, anomaly_count, severity_breakdown, anomalies list, timestamp.
    """
    # Build explanation lookup: match by (type, date, column)
    explanation_map: dict[tuple, dict] = {}
    if explanations:
        for anomaly, explanation in explanations:
            key = (
                anomaly.get("type"),
                str(anomaly.get("date", "")),
                anomaly.get("column"),
            )
            explanation_map[key] = explanation

    severity_breakdown: dict[str, int] = {}
    payload_anomalies = []

    for a in anomalies:
        sev = a.get("severity", "low")
        severity_breakdown[sev] = severity_breakdown.get(sev, 0) + 1

        entry: dict[str, Any] = {
            "type": a.get("type"),
            "severity": sev,
            "date": str(a.get("date", "")),
            "column": a.get("column"),
            "message": a.get("message", ""),
            "expected": a.get("expected"),
            "actual": a.get("actual"),
            "z_score": a.get("z_score"),
        }

        # Attach explanation if available
        key = (a.get("type"), str(a.get("date", "")), a.get("column"))
        if key in explanation_map:
            exp = explanation_map[key]
            entry["root_cause"] = exp.get("root_cause")
            entry["suggested_actions"] = exp.get("suggested_actions")
            entry["business_impact"] = exp.get("business_impact")

        payload_anomalies.append(entry)

    # Build summary string
    parts = []
    for sev in ("critical", "high", "medium", "low"):
        count = severity_breakdown.get(sev, 0)
        if count:
            parts.append(f"{count} {sev}")
    summary = f"{len(anomalies)} anomalies detected: {', '.join(parts)}"

    return {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "anomaly_count": len(anomalies),
        "severity_breakdown": severity_breakdown,
        "anomalies": payload_anomalies,
    }


def send_slack_alert(payload: dict[str, Any], config: dict[str, Any]) -> bool:
    """Send alert to Slack via incoming webhook URL.

    Returns True on success, False on failure.
    """
    webhook_url = config.get("webhook_url", "")
    if not webhook_url:
        print("  Slack: No webhook_url configured, skipping")
        return False

    # Build Slack Block Kit message
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Data Quality Alert"},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*{payload['summary']}*"},
        },
    ]

    # Add top anomalies (limit to 5 for readability)
    for a in payload["anomalies"][:5]:
        severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(
            a["severity"], "⚪"
        )
        text = f"{severity_emoji} *{a['type']}*"
        if a.get("column"):
            text += f" (`{a['column']}`)"
        text += f"\n{a['message']}"
        if a.get("root_cause"):
            text += f"\n_Root cause: {a['root_cause']}_"
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": text}})

    remaining = len(payload["anomalies"]) - 5
    if remaining > 0:
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"_...and {remaining} more anomalies_"}],
        })

    # Add mention if critical
    mention = ""
    if config.get("mention_on_critical") and payload["severity_breakdown"].get("critical", 0) > 0:
        mention = "<!channel> "

    slack_payload = {
        "channel": config.get("channel", "#data-quality-alerts"),
        "text": f"{mention}{payload['summary']}",
        "blocks": blocks,
    }

    try:
        data = json.dumps(slack_payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"  Slack alert failed: {e}")
        return False


def send_email_alert(payload: dict[str, Any], config: dict[str, Any]) -> bool:
    """Send alert via SMTP email.

    Returns True on success, False on failure.
    """
    required = ("smtp_host", "smtp_user", "from_address", "to_addresses")
    for field in required:
        if not config.get(field):
            print(f"  Email: Missing required field '{field}', skipping")
            return False

    if not config["to_addresses"]:
        print("  Email: No to_addresses configured, skipping")
        return False

    # Build HTML email
    rows_html = ""
    for a in payload["anomalies"]:
        color = {"critical": "#dc3545", "high": "#fd7e14", "medium": "#ffc107", "low": "#0d6efd"}.get(
            a["severity"], "#6c757d"
        )
        root_cause = a.get("root_cause", "N/A")
        rows_html += f"""
        <tr>
            <td style="padding:8px;border:1px solid #dee2e6"><span style="color:{color};font-weight:bold">{a['severity'].upper()}</span></td>
            <td style="padding:8px;border:1px solid #dee2e6">{a['type']}</td>
            <td style="padding:8px;border:1px solid #dee2e6">{a.get('column') or '—'}</td>
            <td style="padding:8px;border:1px solid #dee2e6">{a['message']}</td>
            <td style="padding:8px;border:1px solid #dee2e6">{root_cause}</td>
        </tr>"""

    html = f"""
    <html><body>
    <h2>Data Quality Alert</h2>
    <p><strong>{payload['summary']}</strong></p>
    <table style="border-collapse:collapse;width:100%">
        <tr style="background:#f8f9fa">
            <th style="padding:8px;border:1px solid #dee2e6">Severity</th>
            <th style="padding:8px;border:1px solid #dee2e6">Type</th>
            <th style="padding:8px;border:1px solid #dee2e6">Column</th>
            <th style="padding:8px;border:1px solid #dee2e6">Message</th>
            <th style="padding:8px;border:1px solid #dee2e6">Root Cause</th>
        </tr>
        {rows_html}
    </table>
    <p style="color:#6c757d;font-size:12px">Generated at {payload['timestamp']}</p>
    </body></html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Data Quality Alert: {payload['summary']}"
    msg["From"] = config["from_address"]
    msg["To"] = ", ".join(config["to_addresses"])
    msg.attach(MIMEText(payload["summary"], "plain"))
    msg.attach(MIMEText(html, "html"))

    try:
        smtp_port = config.get("smtp_port", 587)
        use_tls = config.get("use_tls", True)

        if use_tls:
            context = ssl.create_default_context()
            server = smtplib.SMTP(config["smtp_host"], smtp_port)
            server.starttls(context=context)
        else:
            server = smtplib.SMTP(config["smtp_host"], smtp_port)

        password = config.get("smtp_password", "")
        if password:
            server.login(config["smtp_user"], password)

        server.sendmail(config["from_address"], config["to_addresses"], msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"  Email alert failed: {e}")
        return False


def send_webhook_alert(payload: dict[str, Any], config: dict[str, Any]) -> bool:
    """Send alert to a generic webhook URL.

    Returns True on success, False on failure.
    """
    url = config.get("url", "")
    if not url:
        print("  Webhook: No url configured, skipping")
        return False

    # Optionally strip explanations
    send_payload = payload
    if not config.get("include_explanations", True):
        send_payload = {**payload}
        send_payload["anomalies"] = [
            {k: v for k, v in a.items() if k not in ("root_cause", "suggested_actions", "business_impact")}
            for a in payload["anomalies"]
        ]

    try:
        data = json.dumps(send_payload, default=str).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        headers.update(config.get("headers", {}))
        method = config.get("method", "POST").upper()

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception as e:
        print(f"  Webhook alert failed: {e}")
        return False


def send_alerts(
    anomalies: list[dict[str, Any]],
    explanations: Optional[list[tuple[dict, dict]]] = None,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, bool]:
    """Filter anomalies, build payload, and send to all enabled channels.

    Args:
        anomalies: All detected anomalies.
        explanations: Optional list of (anomaly, explanation) tuples.
        config: Alert config dict. If None, loads from config/alert_config.json.

    Returns:
        Dict mapping channel name to success/failure boolean.
        Only includes channels that were enabled and attempted.
    """
    if config is None:
        try:
            config = load_alert_config()
        except FileNotFoundError:
            print("  No alert config found, skipping alerts")
            return {}

    if not config.get("enabled", False):
        return {}

    # Filter anomalies
    filtered = filter_anomalies(
        anomalies,
        severity_threshold=config.get("severity_threshold", "medium"),
        anomaly_types=config.get("anomaly_types") or None,
    )

    if not filtered:
        print("  No anomalies match alert filters, skipping alerts")
        return {}

    # Build payload
    payload = build_alert_payload(filtered, explanations)

    print(f"\n  Sending alerts for {len(filtered)} anomalies...")

    # Send to each enabled channel
    results: dict[str, bool] = {}
    channels = config.get("channels", {})

    senders = {
        "slack": send_slack_alert,
        "email": send_email_alert,
        "webhook": send_webhook_alert,
    }

    for channel_name, sender_fn in senders.items():
        channel_config = channels.get(channel_name, {})
        if channel_config.get("enabled", False):
            success = sender_fn(payload, channel_config)
            results[channel_name] = success
            status = "sent" if success else "FAILED"
            print(f"  Alert [{channel_name}]: {status}")

    return results
