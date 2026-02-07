## Project Setup

```bash
cd data-quality-platform
python -m venv venv
source venv/bin/activate
python setup_environment.py
```

This will install all the dependencies from requirements.txt.

## Usage

### Run Full Pipeline (with data generation)

```bash
python pipeline.py --all
```

This generates sample e-commerce data with injected anomalies, profiles it, detects anomalies, and explains them using the configured LLM provider.

### Run Pipeline on Existing Data

```bash
python pipeline.py --no-generate
```

Skips data generation and runs profiling, anomaly detection, and explanation on existing files in the `data/` directory.

### Run Individual Modules

```bash
python -m src.data_generator      # Generate sample data
python -m src.data_profiler       # Profile datasets
python -m src.anomaly_detector    # Detect anomalies
python -m src.llm_explainer       # Explain anomalies with LLM
streamlit run src/dashboard.py    # Launch interactive dashboard
```

### Launch Dashboard

```bash
streamlit run src/dashboard.py
```

Opens an interactive web dashboard with anomaly timeline, severity filters, and AI-powered explanations.

## Modules

| Module | Description |
|--------|-------------|
| `data_generator.py` | Generates realistic e-commerce data with intentional anomalies |
| `data_loader.py` | Universal data loader supporting CSV, Parquet, JSON, Excel, and more |
| `data_profiler.py` | Computes statistical profiles and baselines for datasets |
| `anomaly_detector.py` | Detects 20+ anomaly types using z-score and distribution analysis |
| `llm_explainer.py` | Generates root cause explanations using LLM providers via LiteLLM |
| `dashboard.py` | Streamlit-based interactive monitoring dashboard |

## Custom Data

To analyze your own data, place files in the `data/` directory and run:

```bash
python pipeline.py --no-generate
```

Supported formats: CSV, Parquet, JSON, JSONL, Excel, Feather, HDF5, Pickle.

## LLM Configuration

Configure the LLM provider in `config/llm_config.json`. Supports Anthropic, OpenAI, Azure, Cohere, Gemini, Bedrock, and 100+ providers via LiteLLM. Set `use_mock: true` for demo mode without an API key.
