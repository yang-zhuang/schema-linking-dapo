"""
Utility functions for evaluation system
"""

import re
import json
from typing import Dict, Set, Tuple
import numpy as np


def extract_tables_and_columns(standard_str: str) -> Tuple[Set[str], Set[str]]:
    """
    Extract table names and column names from standardized string
    :param standard_str: Format like "###Tables: 表1,表2;\n###Columns: 表1.列1,表2.列2;"
    :return: (table_name_set, column_name_set)
    """
    # Extract table names
    tables_match = re.search(r'Tables:\s*(.*?);', standard_str)
    tables = set(tables_match.group(1).split(', ')) if (tables_match and tables_match.group(1).strip()) else set()

    # Extract column names
    cols_match = re.search(r'Columns:\s*(.*?);', standard_str)
    columns = set(col.strip() for col in cols_match.group(1).split(', ')) if (
            cols_match and cols_match.group(1).strip()) else set()

    # Convert to lowercase to avoid case sensitivity issues
    return {t.lower() for t in tables}, {c.lower() for c in columns}


def convert_numpy_types(obj):
    """
    Convert numpy types to Python native types for JSON serialization
    """
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj