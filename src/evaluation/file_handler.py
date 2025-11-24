"""
File handling and result saving module
"""

import os
import json
import pandas as pd
from typing import List, Dict

from .utils import convert_numpy_types


class FileHandler:
    """Handles file operations and result saving"""

    def save_process_results(self, results: List[Dict], save_path: str) -> None:
        """
        Save processing results as CSV file (with complete information for error analysis)
        :param results: List of processing results
        :param save_path: Save path
        """
        try:
            # 1. Handle JSON type fields (convert to strings to avoid CSV format issues)
            results_for_save = []
            for res in results:
                # Deep copy to avoid modifying original data
                res_copy = res.copy()
                # Convert JSON objects to formatted strings
                res_copy["pred_json"] = json.dumps(res_copy["pred_json"], ensure_ascii=False, indent=2)
                res_copy["ground_truth_json"] = json.dumps(res_copy["ground_truth_json"], ensure_ascii=False, indent=2)
                results_for_save.append(res_copy)

            # 2. Convert to DataFrame and save
            save_df = pd.DataFrame(results_for_save)
            # Create save directory (if not exists)
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Save CSV (index=False to avoid extra row numbers, encoding=utf-8 to support Chinese)
            save_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"\n✅ Processing results saved to: {os.path.abspath(save_path)}")

        except Exception as e:
            # Catch save exceptions, don't interrupt main flow
            print(f"\n⚠️  Result saving failed: {str(e)}")

    def save_evaluation_metrics(self, metrics: Dict, metrics_path: str) -> None:
        """
        Save evaluation metrics as JSON file (for subsequent analysis and visualization)
        :param metrics: Evaluation metrics dictionary
        :param metrics_path: Metrics save path
        """
        try:
            # Create save directory (if not exists)
            metrics_dir = os.path.dirname(metrics_path)
            if not os.path.exists(metrics_dir):
                os.makedirs(metrics_dir)

            # Convert numpy types to Python native types to ensure JSON serializability
            metrics_serializable = convert_numpy_types(metrics)

            # Save as JSON file
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_serializable, f, ensure_ascii=False, indent=2)

            print(f"✅ Evaluation metrics saved to: {os.path.abspath(metrics_path)}")

        except Exception as e:
            print(f"⚠️  Evaluation metrics saving failed: {str(e)}")