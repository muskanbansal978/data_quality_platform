"""
Universal Data Loader for multiple file formats.

Supports: CSV, Parquet, JSON, Excel, Feather, HDF5, Pickle
"""

import pandas as pd
from pathlib import Path
from typing import Union


class UniversalDataLoader:
    """Loads data from various file formats."""

    SUPPORTED_FORMATS = {
        '.csv': 'CSV',
        '.parquet': 'Parquet',
        '.pq': 'Parquet',
        '.json': 'JSON',
        '.jsonl': 'JSON Lines',
        '.xlsx': 'Excel',
        '.xls': 'Excel',
        '.feather': 'Feather',
        '.h5': 'HDF5',
        '.hdf5': 'HDF5',
        '.pkl': 'Pickle',
        '.pickle': 'Pickle',
    }

    def load(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from any supported file format.

        Args:
            file_path: Path to the file

        Returns:
            DataFrame with loaded data
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {ext}")

        # Load based on extension
        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext in ['.parquet', '.pq']:
            return pd.read_parquet(file_path)
        elif ext == '.json':
            return pd.read_json(file_path)
        elif ext == '.jsonl':
            return pd.read_json(file_path, lines=True)
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif ext == '.feather':
            return pd.read_feather(file_path)
        elif ext in ['.h5', '.hdf5']:
            # Try to read the first key
            with pd.HDFStore(file_path, 'r') as store:
                keys = store.keys()
                if keys:
                    return pd.read_hdf(file_path, key=keys[0])
                raise ValueError("HDF5 file is empty")
        elif ext in ['.pkl', '.pickle']:
            return pd.read_pickle(file_path)


def get_file_info(file_path: Union[str, Path]) -> dict:
    """Get information about a data file."""
    file_path = Path(file_path)

    info = {
        'name': file_path.name,
        'extension': file_path.suffix.lower(),
        'size_mb': file_path.stat().st_size / (1024 * 1024),
        'supported': file_path.suffix.lower() in UniversalDataLoader.SUPPORTED_FORMATS,
    }

    if info['supported']:
        info['format'] = UniversalDataLoader.SUPPORTED_FORMATS[info['extension']]

    return info
