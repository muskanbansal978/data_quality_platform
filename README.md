## Project Setup ##
cd data-quality-platform
python -m venv venv
source venv/bin/activate
pip install -e . 
[Note] This will install all the dependencies mentioned in pyproject.toml

## Implementation ##
data_generator.py is used to create sample data
data_profile.py is used for computing statistical profiles of datasets
anomaly_detector.py is used for data quality monitoring

If you want a different data to be profiled add in .csv format in data dir in root directory


