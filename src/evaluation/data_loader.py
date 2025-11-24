"""
Data loading and processing module
"""

import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple

from .config import DEFAULT_SYSTEM_PROMPT
from .utils import extract_tables_and_columns


class DataLoader:
    """Handles data loading and processing"""

    def load_input_data(self, file_path: str) -> pd.DataFrame:
        """Load input CSV data with validation"""
        try:
            df = pd.read_csv(file_path)
            # Validate required columns
            required_cols = ["question", "database_schema", "target_schema"]
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV missing required columns: {required_cols}")
            return df
        except Exception as e:
            raise ValueError(f"Data loading failed: {str(e)}")

    def process_ground_truth_schema(self, target_schema_raw: str) -> Tuple[Dict[str, List[Dict]], str]:
        """
        Process ground truth schema: convert raw string to standard JSON and standardized string
        :param target_schema_raw: Raw label string (e.g., "###Tables: singer;\n###Columns: singer.name;")
        :return: (ground truth JSON, ground truth standardized string)
        """
        # Extract table names and column names
        truth_tables, truth_cols = extract_tables_and_columns(target_schema_raw)

        # Build ground truth JSON (aligned with model output format)
        ground_truth_json = {"schema": []}
        table_col_map = {table: [] for table in truth_tables}
        for col in truth_cols:
            try:
                table, col_name = col.split(".")
                if table in table_col_map:
                    table_col_map[table].append(col_name)
            except ValueError:
                continue  # Skip malformed columns

        # Fill JSON structure
        for table, cols in table_col_map.items():
            ground_truth_json["schema"].append({"table_name": table, "columns": cols})

        # Generate standardized string (for comparison)
        tables_str = ', '.join(truth_tables)
        cols_str = ', '.join(truth_cols)
        ground_truth_str = f"###Tables: {tables_str};\n###Columns: {cols_str};"

        return ground_truth_json, ground_truth_str

    def build_model_prompt(self, question: str, db_schema: str) -> str:
        """Build model input prompt"""
        return f"<question>{question}</question>\n<database>{db_schema}</database>"

    def prepare_data(self, df: pd.DataFrame, debug: bool = False, debug_samples: int = 50) -> Tuple[List[str], List[Dict]]:
        """
        Prepare data for evaluation
        :param df: Input DataFrame
        :param debug: Whether to use debug mode
        :param debug_samples: Number of samples for debug mode
        :return: (prompts list, ground truth data list)
        """
        total = min(debug_samples, len(df)) if debug else len(df)

        prompts = []
        ground_truth_data = []

        print("Preparing data...")
        for idx, row in tqdm(df.head(total).iterrows(), total=total, desc="Data preparation"):
            # Process ground truth
            ground_truth_json, ground_truth_str = self.process_ground_truth_schema(row["target_schema"])

            # Build prompt
            model_prompt = self.build_model_prompt(row["question"], row["database_schema"])
            prompts.append(model_prompt)

            ground_truth_data.append({
                "question": row["question"],
                "ground_truth_json": ground_truth_json,
                "ground_truth_standard_str": ground_truth_str
            })

        return prompts, ground_truth_data