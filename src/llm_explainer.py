"""
LLM-powered anomaly explainer with universal provider support.

Uses any LLM provider (Anthropic, OpenAI, Azure, Cohere, Gemini, Bedrock, etc.)
to generate natural language explanations, root cause hypotheses, and remediation
suggestions for detected anomalies.

Leverages LiteLLM for unified access to 100+ LLM providers.

Run with: python -m src.llm_explainer

Configuration: Requires llm_config.json in config folder with provider, model, and API key
"""
 
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Default config file path
CONFIG_FILE = Path(__file__).resolve().parents[1] / "config" / "llm_config.json"


def load_llm_config() -> dict[str, Any]:
    """Load LLM configuration from config file."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
    
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    
    return config


def get_provider_and_model() -> tuple[str, str, bool, Optional[str], Optional[dict]]:
    """Get provider, model, use_mock flag, API key, and additional config from config.

    Returns:
        Tuple of (provider, model, use_mock, api_key, custom_llm_provider_config)
    """
    config = load_llm_config()

    provider = config.get("provider")
    if not provider:
        raise ValueError("'provider' must be specified in llm_config.json")
    provider = provider.lower()

    use_mock = config.get("use_mock", False)

    # Get model from config, with fallback to provider-specific config
    if config.get("model"):
        model = config["model"]
    else:
        model = config.get(provider, {}).get("model")

    if not model:
        raise ValueError(f"'model' must be specified in llm_config.json for provider '{provider}'")

    # Get API key from provider-specific config or top-level
    api_key = config.get(provider, {}).get("api_key") or config.get("api_key") or None

    # Get additional provider-specific configuration (e.g., base_url, api_version)
    custom_llm_provider_config = config.get(provider, {}).copy()
    custom_llm_provider_config.pop("api_key", None)  # Remove API key from custom config
    custom_llm_provider_config.pop("model", None)  # Remove model from custom config

    return provider, model, use_mock, api_key, custom_llm_provider_config


# System prompt for LLM
SYSTEM_PROMPT = """You are a data quality expert analyzing anomalies in data pipelines.
Your job is to:
1. Identify the most likely root cause of each anomaly
2. Suggest specific remediation actions
3. Assess business impact

Be concise but specific. Focus on actionable insights.
Base your analysis on the data patterns and context provided.

Always respond in valid JSON format with this structure:
{
    "root_cause": "Brief explanation of the most likely cause",
    "confidence": "high|medium|low",
    "suggested_actions": ["Action 1", "Action 2", "Action 3"],
    "business_impact": "Brief assessment of business impact",
    "additional_context": "Any other relevant observations (optional)"
}"""

# Mock explanations for testing without API
MOCK_EXPLANATIONS = {
    # Volume anomalies
    "volume_drop": {
        "root_cause": "Likely caused by an upstream data pipeline failure or source system outage. The 50% drop suggests a complete failure rather than partial data loss.",
        "confidence": "high",
        "suggested_actions": [
            "Check upstream pipeline job logs for failures around the affected date",
            "Verify source system availability during the time window",
            "Review any scheduled maintenance that may have occurred",
            "Set up monitoring alerts for volume drops > 20%",
        ],
        "business_impact": "HIGH - Missing data may affect daily reports and downstream analytics",
        "additional_context": None,
    },
    "volume_spike": {
        "root_cause": "Unusual spike in data volume. Could indicate duplicate data loading, a backfill operation, or genuinely increased activity.",
        "confidence": "medium",
        "suggested_actions": [
            "Check for duplicate records using primary key analysis",
            "Review ETL job history for any backfill operations",
            "Verify if this correlates with any business events (promotions, etc.)",
        ],
        "business_impact": "MEDIUM - May cause inflated metrics if duplicates exist",
        "additional_context": None,
    },
    # Null/completeness anomalies
    "null_rate_increase": {
        "root_cause": "Significant increase in null values suggests a schema change, API contract violation, or data source issue. The sudden jump indicates a discrete change rather than gradual degradation.",
        "confidence": "high",
        "suggested_actions": [
            "Review recent schema changes in source systems",
            "Check API response logs for the affected field",
            "Validate data extraction logic for the column",
            "Add null rate monitoring with alerting threshold",
        ],
        "business_impact": "MEDIUM - Missing customer IDs may affect attribution and analytics",
        "additional_context": None,
    },
    # Value range anomalies
    "negative_values": {
        "root_cause": "Negative values in a typically positive field often indicates refunds, chargebacks, or corrections being mixed with regular transactions. This could also be a sign of sign-flip bugs in calculations.",
        "confidence": "high",
        "suggested_actions": [
            "Identify the source of negative values (refunds vs bugs)",
            "Check if transaction type field should be used to separate refunds",
            "Add data validation to flag or reject unexpected negative values",
            "Review recent ETL changes that might have affected sign handling",
        ],
        "business_impact": "HIGH - Affects revenue calculations and financial reporting accuracy",
        "additional_context": None,
    },
    "extreme_value": {
        "root_cause": "Extreme values beyond historical bounds suggest data entry errors, unit conversion issues, or edge cases not previously encountered.",
        "confidence": "medium",
        "suggested_actions": [
            "Identify specific records with extreme values",
            "Check for unit conversion errors (e.g., cents vs dollars)",
            "Review data entry validation rules",
            "Consider if business rules allow these edge cases",
        ],
        "business_impact": "MEDIUM - May skew aggregations and statistical analyses",
        "additional_context": None,
    },
    # Statistical anomalies
    "mean_shift": {
        "root_cause": "Significant shift in average values indicates either a change in data composition, pricing changes, or calculation errors. Could also reflect genuine business changes.",
        "confidence": "medium",
        "suggested_actions": [
            "Compare data distribution before and after the shift",
            "Check for changes in product mix or category distribution",
            "Review any pricing or business rule changes around this date",
            "Validate calculations in the ETL pipeline",
        ],
        "business_impact": "MEDIUM - May affect trend analysis and forecasting accuracy",
        "additional_context": None,
    },
    "median_shift": {
        "root_cause": "Median shift indicates a fundamental change in the central tendency of data, more robust to outliers than mean. Could indicate pricing changes, customer behavior shifts, or data source changes.",
        "confidence": "high",
        "suggested_actions": [
            "Analyze distribution changes using percentile comparisons",
            "Check for changes in customer segments or product mix",
            "Review any business policy changes affecting this metric",
            "Compare with related metrics to identify patterns",
        ],
        "business_impact": "MEDIUM - Affects KPIs that rely on typical values",
        "additional_context": None,
    },
    "variance_shift": {
        "root_cause": "Change in data variance/volatility indicates either more diverse data, inconsistent data quality, or changes in measurement precision.",
        "confidence": "medium",
        "suggested_actions": [
            "Analyze if increased variance is due to outliers or general spread",
            "Check for changes in data collection methodology",
            "Review if multiple data sources are being merged differently",
            "Verify measurement precision hasn't changed",
        ],
        "business_impact": "LOW - May affect confidence intervals and statistical tests",
        "additional_context": None,
    },
    "distribution_shift": {
        "root_cause": "Category distribution change suggests either a change in business patterns, data collection changes, or potential data quality issues in categorization.",
        "confidence": "medium",
        "suggested_actions": [
            "Verify categorization logic hasn't changed",
            "Check if this correlates with marketing campaigns or seasonality",
            "Compare with historical seasonal patterns",
            "Review any changes to product catalog or category mappings",
        ],
        "business_impact": "LOW - May affect segment-level reporting but overall totals likely unaffected",
        "additional_context": None,
    },
    "outlier_spike": {
        "root_cause": "Sudden increase in outlier count suggests batch data quality issues, system glitches, or unusual business events affecting multiple records.",
        "confidence": "high",
        "suggested_actions": [
            "Identify common attributes of outlier records",
            "Check for system issues during data collection window",
            "Review if outliers share a common source or batch",
            "Consider adjusting outlier detection thresholds if business-justified",
        ],
        "business_impact": "MEDIUM - Outliers may distort aggregations and ML models",
        "additional_context": None,
    },
    # Cardinality anomalies
    "cardinality_increase": {
        "root_cause": "Increase in unique values could indicate new products/categories, data quality issues (typos creating new values), or expanded data coverage.",
        "confidence": "medium",
        "suggested_actions": [
            "Identify the new unique values that appeared",
            "Check if new values are legitimate additions or data errors",
            "Review data entry forms for validation issues",
            "Update reference data if new values are valid",
        ],
        "business_impact": "LOW - May require updates to dashboards and reports",
        "additional_context": None,
    },
    "cardinality_decrease": {
        "root_cause": "Decrease in unique values suggests data filtering, category consolidation, or missing data segments.",
        "confidence": "medium",
        "suggested_actions": [
            "Identify which values disappeared",
            "Check for unintended data filtering in ETL",
            "Verify all data sources are being captured",
            "Review if business discontinued certain categories",
        ],
        "business_impact": "MEDIUM - May indicate missing data segments",
        "additional_context": None,
    },
    "new_category": {
        "root_cause": "A new categorical value appeared that wasn't seen historically. Could be a legitimate new category or a data entry error.",
        "confidence": "low",
        "suggested_actions": [
            "Verify if the new value is a valid addition",
            "Check for typos or encoding issues",
            "Update validation rules to include new valid values",
            "Add the new category to reference data if legitimate",
        ],
        "business_impact": "LOW - May need reporting adjustments",
        "additional_context": None,
    },
    "missing_category": {
        "root_cause": "A previously seen category value is now missing. Could indicate data filtering issues or genuine business changes.",
        "confidence": "low",
        "suggested_actions": [
            "Check if the category was intentionally deprecated",
            "Verify ETL is capturing all source data",
            "Review filter conditions in data pipelines",
            "Confirm with business if category discontinuation is expected",
        ],
        "business_impact": "LOW - May affect historical comparisons",
        "additional_context": None,
    },
    # Duplicate anomalies
    "duplicate_spike": {
        "root_cause": "Spike in duplicate records typically indicates ETL job re-runs, source system issues, or missing deduplication logic.",
        "confidence": "high",
        "suggested_actions": [
            "Check ETL job execution logs for re-runs",
            "Review deduplication logic in the pipeline",
            "Verify source system isn't sending duplicate events",
            "Add idempotency checks to prevent future duplicates",
        ],
        "business_impact": "HIGH - Duplicates inflate metrics and affect data accuracy",
        "additional_context": None,
    },
    "duplicate_key_violation": {
        "root_cause": "Duplicate primary/unique key values indicate data integrity issues in the source or ETL process.",
        "confidence": "high",
        "suggested_actions": [
            "Identify records with duplicate keys",
            "Review key generation logic",
            "Add unique constraints to catch issues earlier",
            "Implement merge/upsert logic instead of insert",
        ],
        "business_impact": "HIGH - Breaks data integrity and joins",
        "additional_context": None,
    },
    # Freshness anomalies
    "data_staleness": {
        "root_cause": "Data hasn't been updated recently. Could indicate pipeline failure, source system issues, or scheduling problems.",
        "confidence": "high",
        "suggested_actions": [
            "Check pipeline scheduler for failed or stuck jobs",
            "Verify source system is operational",
            "Review any recent infrastructure changes",
            "Set up freshness monitoring alerts",
        ],
        "business_impact": "HIGH - Stale data leads to outdated decisions",
        "additional_context": None,
    },
    "future_date": {
        "root_cause": "Data contains timestamps in the future, indicating timezone issues, clock synchronization problems, or data entry errors.",
        "confidence": "high",
        "suggested_actions": [
            "Review timezone handling in the pipeline",
            "Check source system clock synchronization",
            "Validate timestamp generation logic",
            "Add validation to reject future dates",
        ],
        "business_impact": "MEDIUM - Affects time-based queries and reports",
        "additional_context": None,
    },
    "late_arriving_data": {
        "root_cause": "Data arrived much later than expected, which may affect real-time analytics and timely reporting.",
        "confidence": "medium",
        "suggested_actions": [
            "Review data latency from source systems",
            "Check for batching delays in the pipeline",
            "Consider implementing late-arriving data handling",
            "Adjust SLAs if delays are expected",
        ],
        "business_impact": "MEDIUM - Affects real-time dashboards and alerts",
        "additional_context": None,
    },
    # String anomalies
    "string_length_anomaly": {
        "root_cause": "Change in average string length suggests schema changes, truncation issues, or different data formats being ingested.",
        "confidence": "medium",
        "suggested_actions": [
            "Check for field truncation in source or ETL",
            "Review if data format changed (e.g., full names vs initials)",
            "Verify varchar limits aren't cutting off data",
            "Compare sample values before and after the change",
        ],
        "business_impact": "LOW - May affect display and search functionality",
        "additional_context": None,
    },
    "whitespace_anomaly": {
        "root_cause": "Increase in values with leading/trailing whitespace indicates data quality issues in source entry or ETL transformations.",
        "confidence": "high",
        "suggested_actions": [
            "Add TRIM operations to ETL pipeline",
            "Review source data entry forms",
            "Clean existing data with whitespace issues",
            "Add validation to prevent whitespace on entry",
        ],
        "business_impact": "LOW - Affects matching and joins",
        "additional_context": None,
    },
    "pattern_violation": {
        "root_cause": "Values don't match expected format patterns (e.g., email, phone). Could indicate validation bypass or data migration issues.",
        "confidence": "high",
        "suggested_actions": [
            "Identify records with pattern violations",
            "Review input validation in source systems",
            "Check for data migration or import issues",
            "Add format validation in ETL pipeline",
        ],
        "business_impact": "MEDIUM - Invalid formats may break downstream systems",
        "additional_context": None,
    },
    # Schema anomalies
    "schema_drift": {
        "root_cause": "Schema changed unexpectedly - columns added or removed. Could indicate source system updates or ETL configuration changes.",
        "confidence": "high",
        "suggested_actions": [
            "Review source system release notes",
            "Update ETL to handle new schema",
            "Verify if column removal was intentional",
            "Add schema validation to detect drift early",
        ],
        "business_impact": "HIGH - May break downstream pipelines and reports",
        "additional_context": None,
    },
    "type_mismatch": {
        "root_cause": "Column data type changed, which can cause parsing errors and calculation issues.",
        "confidence": "high",
        "suggested_actions": [
            "Review source schema changes",
            "Update ETL type casting logic",
            "Validate data can be safely converted",
            "Add type checking in pipeline",
        ],
        "business_impact": "HIGH - Type mismatches can cause pipeline failures",
        "additional_context": None,
    },
    # Correlation anomalies
    "correlation_break": {
        "root_cause": "Historical correlation between columns has broken, suggesting calculation errors, business rule changes, or data quality issues in one of the fields.",
        "confidence": "medium",
        "suggested_actions": [
            "Verify calculation logic for both columns",
            "Check if business rules changed affecting the relationship",
            "Look for data quality issues in either column",
            "Review if the correlation was coincidental",
        ],
        "business_impact": "MEDIUM - May indicate data integrity issues",
        "additional_context": None,
    },
    # Sequence anomalies
    "sequence_gap": {
        "root_cause": "Gaps in sequential IDs suggest missing records, failed transactions, or ID generation issues.",
        "confidence": "medium",
        "suggested_actions": [
            "Identify the range of missing IDs",
            "Check for failed transactions or rollbacks",
            "Review ID generation mechanism",
            "Verify if gaps are expected (e.g., reservations)",
        ],
        "business_impact": "MEDIUM - May indicate missing data",
        "additional_context": None,
    },
    "out_of_order": {
        "root_cause": "Records are out of chronological order, which may indicate late-arriving data or timestamp issues.",
        "confidence": "medium",
        "suggested_actions": [
            "Review timestamp generation in source system",
            "Check for timezone inconsistencies",
            "Implement ordering guarantees in pipeline",
            "Consider event time vs processing time",
        ],
        "business_impact": "LOW - May affect time-series analysis",
        "additional_context": None,
    },
}


def create_explanation(
    root_cause: str,
    confidence: str,
    suggested_actions: list[str],
    business_impact: str,
    additional_context: Optional[str] = None,
) -> dict[str, Any]:
    """Create an explanation dictionary."""
    return {
        "root_cause": root_cause,
        "confidence": confidence,
        "suggested_actions": suggested_actions,
        "business_impact": business_impact,
        "additional_context": additional_context,
    }


class UniversalLLMProvider:
    """Universal LLM provider using LiteLLM to support all providers dynamically."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        custom_config: Optional[dict] = None
    ):
        """
        Initialize universal LLM provider.

        Args:
            provider: Provider name (e.g., 'anthropic', 'openai', 'azure', 'cohere', etc.)
            model: Model name for the provider
            api_key: API key for authentication
            custom_config: Additional provider-specific configuration (e.g., base_url, api_version)
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.custom_config = custom_config or {}
        self._initialized = False

    def initialize_client(self):
        """Initialize and verify LiteLLM can be imported."""
        if self._initialized:
            return

        try:
            import litellm
            self._initialized = True
        except ImportError:
            raise ImportError(
                "litellm package not installed. "
                "Run: pip install litellm"
            )

    def call(self, prompt: str, system_prompt: str) -> str:
        """
        Call the LLM using LiteLLM's unified interface.

        LiteLLM automatically handles different provider formats and requirements.
        """
        self.initialize_client()

        import litellm

        # Build messages in the standard format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Prepare kwargs for litellm.completion
        kwargs = {
            "model": f"{self.provider}/{self.model}",
            "messages": messages,
            "max_tokens": 1024,
        }

        # Add API key if provided
        if self.api_key:
            kwargs["api_key"] = self.api_key

        # Merge any custom configuration (e.g., base_url, api_version, temperature)
        kwargs.update(self.custom_config)

        try:
            response = litellm.completion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            # Provide helpful error message
            raise RuntimeError(
                f"LLM call failed for provider '{self.provider}' with model '{self.model}': {str(e)}"
            )


def get_llm_provider(
    provider_name: str,
    model: str,
    api_key: Optional[str] = None,
    custom_config: Optional[dict] = None
) -> UniversalLLMProvider:
    """
    Factory function to get a universal LLM provider.

    Supports all providers through LiteLLM including:
    - anthropic (Claude)
    - openai (GPT)
    - azure (Azure OpenAI)
    - cohere (Command models)
    - gemini (Google Gemini)
    - bedrock (AWS Bedrock)
    - replicate
    - huggingface
    - And 100+ more providers

    Args:
        provider_name: Name of the LLM provider
        model: Model identifier
        api_key: API key for authentication
        custom_config: Additional provider-specific configuration

    Returns:
        UniversalLLMProvider instance
    """
    return UniversalLLMProvider(provider_name, model, api_key, custom_config)


def build_prompt(
    anomaly: dict[str, Any],
    context: Optional[dict[str, Any]] = None,
) -> str:
    """Build the analysis prompt for the LLM."""
    prompt_parts = [
        "Analyze this data quality anomaly:\n",
        f"**Anomaly Type:** {anomaly['type']}",
        f"**Severity:** {anomaly['severity']}",
        f"**Date:** {anomaly['date']}",
        f"**Column:** {anomaly.get('column', 'N/A')}",
        f"**Message:** {anomaly['message']}",
        f"**Expected Value:** {anomaly['expected']}",
        f"**Actual Value:** {anomaly['actual']}",
        f"**Z-Score:** {anomaly['z_score']}",
    ]

    if context:
        prompt_parts.append("\n**Additional Context:**")

        if "table_name" in context:
            prompt_parts.append(f"- Table: {context['table_name']}")

        if "column_description" in context:
            prompt_parts.append(f"- Column description: {context['column_description']}")

        if "historical_pattern" in context:
            prompt_parts.append(f"- Historical pattern: {context['historical_pattern']}")

        if "related_changes" in context:
            prompt_parts.append(f"- Recent changes: {context['related_changes']}")

        if "data_sample" in context:
            prompt_parts.append(f"- Sample values: {context['data_sample']}")

    prompt_parts.append("\nProvide your analysis in JSON format.")

    return "\n".join(prompt_parts)


def parse_response(response_text: str) -> dict[str, Any]:
    """Parse LLM response into structured explanation."""
    # Clean up response - handle markdown code blocks
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
        return create_explanation(
            root_cause=data.get("root_cause", "Unable to determine root cause"),
            confidence=data.get("confidence", "low"),
            suggested_actions=data.get("suggested_actions", []),
            business_impact=data.get("business_impact", "Unknown"),
            additional_context=data.get("additional_context"),
        )
    except json.JSONDecodeError:
        # Fallback if response isn't valid JSON
        return create_explanation(
            root_cause=response_text[:500],
            confidence="low",
            suggested_actions=["Review the anomaly manually"],
            business_impact="Unable to assess",
            additional_context="LLM response was not in expected format",
        )


def explain_anomaly(
    anomaly: dict[str, Any],
    context: Optional[dict[str, Any]] = None,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    custom_config: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Generate explanation for a single anomaly using any LLM provider.

    Args:
        anomaly: Anomaly dict with type, severity, date, column, message, etc.
        context: Optional additional context (table name, descriptions, etc.)
        provider_name: LLM provider name (e.g., 'anthropic', 'openai', 'azure', 'cohere', 'gemini', etc.)
        model: Model name for the provider
        api_key: API key for the provider (optional, uses config or environment)
        custom_config: Additional provider-specific configuration (e.g., base_url, api_version, temperature)

    Returns:
        Explanation dict with root cause, actions, and impact assessment
    """
    if not provider_name or not model:
        raise ValueError("Both provider_name and model must be specified")

    prompt = build_prompt(anomaly, context)

    # Get provider and call
    provider = get_llm_provider(provider_name, model, api_key, custom_config)
    response = provider.call(prompt, SYSTEM_PROMPT)

    return parse_response(response)


def explain_anomaly_mock(anomaly: dict[str, Any]) -> dict[str, Any]:
    """Return mock explanation based on anomaly type (for testing without API)."""
    anomaly_type = anomaly.get("type", "").lower()

    # Find matching mock explanation
    for key, explanation in MOCK_EXPLANATIONS.items():
        if key in anomaly_type:
            return explanation.copy()

    # Default fallback
    return create_explanation(
        root_cause="Anomaly detected but requires manual investigation to determine root cause.",
        confidence="low",
        suggested_actions=[
            "Review the data manually around the affected date",
            "Check for any system changes or deployments",
            "Consult with data source owners",
        ],
        business_impact="Unknown - requires investigation",
    )


def explain_batch(
    anomalies: list[dict[str, Any]],
    context: Optional[dict[str, Any]] = None,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    use_mock: Optional[bool] = None,
    api_key: Optional[str] = None,
    custom_config: Optional[dict] = None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """
    Generate explanations for multiple anomalies using any LLM provider.

    Args:
        anomalies: List of anomaly dicts
        context: Optional shared context
        provider_name: LLM provider name (optional, uses config if not specified)
        model: Model name (optional, uses config if not specified)
        use_mock: Use mock explanations (optional, uses config if not specified)
        api_key: API key for the provider (optional, uses config if not specified)
        custom_config: Additional provider-specific configuration (optional, uses config if not specified)

    Returns:
        List of (anomaly, explanation) tuples
    """
    results = []

    # Load from config if not provided
    if (provider_name is None or model is None or use_mock is None or
        api_key is None or custom_config is None):
        config_provider, config_model, config_use_mock, config_api_key, config_custom = get_provider_and_model()
        if provider_name is None:
            provider_name = config_provider
        if model is None:
            model = config_model
        if use_mock is None:
            use_mock = config_use_mock
        if api_key is None:
            api_key = config_api_key
        if custom_config is None:
            custom_config = config_custom

    # Verify provider can be initialized if not using mock
    if not use_mock:
        try:
            provider = get_llm_provider(provider_name, model, api_key, custom_config)
            provider.initialize_client()
        except (ValueError, ImportError) as e:
            print(f"Warning: Could not initialize LLM provider: {e}")
            print("Falling back to mock explanations.")
            use_mock = True

    for i, anomaly in enumerate(anomalies):
        print(f"Analyzing anomaly {i+1}/{len(anomalies)}...")

        try:
            if use_mock:
                explanation = explain_anomaly_mock(anomaly)
            else:
                explanation = explain_anomaly(anomaly, context, provider_name, model, api_key, custom_config)
            results.append((anomaly, explanation))
        except Exception as e:
            print(f"Error explaining anomaly: {e}")
            results.append((anomaly, create_explanation(
                root_cause=f"Error: {str(e)}",
                confidence="low",
                suggested_actions=["Review manually"],
                business_impact="Unknown",
            )))

    return results


def print_explanation(anomaly: dict[str, Any], explanation: dict[str, Any]) -> None:
    """Pretty print an anomaly with its explanation."""
    severity = anomaly.get("severity", "medium")
    column_info = f" [{anomaly.get('column')}]" if anomaly.get('column') else ""

    print()
    print(f"[{severity.upper()}] {anomaly['type']}{column_info}")
    print(f"  {anomaly['message']}")
    print()
    print("  AI Analysis:")
    print("  " + "-" * 50)
    print(f"  Root Cause ({explanation['confidence']} confidence):")
    print(f"    {explanation['root_cause']}")
    print()
    print("  Suggested Actions:")
    for action in explanation['suggested_actions']:
        print(f"    - {action}")
    print()
    print(f"  Business Impact: {explanation['business_impact']}")

    if explanation.get('additional_context'):
        print(f"  Note: {explanation['additional_context']}")

    print("  " + "-" * 50)


def main():
    """Run LLM explainer on detected anomalies."""
    import pandas as pd
    from src.anomaly_detector import (
        compute_daily_profiles,
        detect_all_anomalies,
        detect_date_column,
    )

    data_dir = Path("data")

    print("\n" + "=" * 50)
    print("        LLM Anomaly Explainer")
    print("=" * 50)

    # Load configuration
    try:
        provider, model, use_mock, api_key, custom_config = get_provider_and_model()
        print(f"\nConfiguration loaded from {CONFIG_FILE}")
        print(f"  Provider: {provider}")
        print(f"  Model: {model}")
        print(f"  Use Mock: {use_mock}")
        if api_key:
            print(f"  API Key: {'*' * (len(api_key) - 4) + api_key[-4:]}")
        if custom_config:
            print(f"  Custom Config: {list(custom_config.keys())}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please create llm_config.json in the project root.")
        return
    except ValueError as e:
        print(f"Error in configuration: {e}")
        return

    # Find CSV files
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        print("\nNo CSV files found in data/ directory.")
        print("Run data_generator first:")
        print("  python -m src.data_generator")
        return

    # Collect all anomalies
    all_anomalies = []

    for file_path in csv_files:
        print(f"\nLoading {file_path.name}...")
        df = pd.read_csv(file_path)

        date_column = detect_date_column(df)
        if date_column is None:
            print(f"  No date column found, skipping")
            continue

        profiles = compute_daily_profiles(df, date_column)
        if len(profiles) < 5:
            print(f"  Not enough history, skipping")
            continue

        anomalies = detect_all_anomalies(profiles)
        for a in anomalies:
            a["source_file"] = file_path.name
        all_anomalies.extend(anomalies)

    print(f"\nFound {len(all_anomalies)} anomalies to explain")

    if not all_anomalies:
        print("No anomalies to explain.")
        return

    if not use_mock:
        # Check for API keys
        api_key = os.environ.get("ANTHROPIC_API_KEY") if provider == "anthropic" else os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print(f"\nWarning: {provider.upper()} API key not found in environment.")
            print(f"Set {provider.upper()}_API_KEY environment variable.")
            print("Using mock explainer for demo.")
            use_mock = True
    
    if use_mock:
        print("\nUsing mock explainer for demo.")
        print("To use real LLM explanations, set API key and update llm_config.json:")
        if provider == "anthropic":
            print("  export ANTHROPIC_API_KEY=your-key-here")
        else:
            print("  export OPENAI_API_KEY=your-key-here")

    # Explain anomalies (limit to first 5 for demo)
    anomalies_to_explain = all_anomalies[:5]

    print(f"\nGenerating explanations for {len(anomalies_to_explain)} anomalies...")
    print("=" * 60)

    for anomaly in anomalies_to_explain:
        if use_mock:
            explanation = explain_anomaly_mock(anomaly)
        else:
            explanation = explain_anomaly(
                anomaly,
                provider_name=provider,
                model=model,
                api_key=api_key,
                custom_config=custom_config
            )

        print_explanation(anomaly, explanation)

    print("\n" + "=" * 60)
    print(f"\nExplained {len(anomalies_to_explain)} anomalies")


if __name__ == "__main__":
    main()
