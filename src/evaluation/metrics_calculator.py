"""
Metrics calculation module
"""

import numpy as np
from typing import Dict, List, Tuple

from .utils import extract_tables_and_columns


class MetricsCalculator:
    """Handles evaluation metrics calculation"""

    def calculate_partial_accuracy(self, ground_truth_str: str, pred_str: str) -> Tuple[float, float]:
        """
        Calculate partial match accuracy (count as correct if prediction contains partial content of ground truth)
        :return: (table partial accuracy, column partial accuracy)
        """
        # Extract tables and columns from ground truth and prediction
        truth_tables, truth_cols = extract_tables_and_columns(ground_truth_str)
        pred_tables, pred_cols = extract_tables_and_columns(pred_str)

        # Calculate table accuracy (avoid division by zero)
        table_correct = len(truth_tables & pred_tables)

        if truth_tables:
            table_acc = table_correct / len(truth_tables)
        else:
            table_acc = 1.0

        # Calculate column accuracy
        col_correct = len(truth_cols & pred_cols)

        if truth_cols:
            col_acc = col_correct / len(truth_cols)
        else:
            col_acc = 1.0

        return table_acc, col_acc

    def calculate_strict_accuracy(self, ground_truth_str: str, pred_str: str) -> Tuple[int, int]:
        """
        Calculate strict match accuracy (only count as correct if prediction exactly matches ground truth)
        :return: (table strict accuracy [0/1], column strict accuracy [0/1])
        """
        truth_tables, truth_cols = extract_tables_and_columns(ground_truth_str)
        pred_tables, pred_cols = extract_tables_and_columns(pred_str)

        table_strict_acc = 1 if (pred_tables == truth_tables) else 0
        col_strict_acc = 1 if (pred_cols == truth_cols) else 0

        return table_strict_acc, col_strict_acc

    def calculate_precision_recall(self, ground_truth_str: str, pred_str: str) -> Tuple[float, float, float, float]:
        """
        Calculate precision and recall for tables and columns
        :return: (table_precision, table_recall, column_precision, column_recall)
        """
        truth_tables, truth_cols = extract_tables_and_columns(ground_truth_str)
        pred_tables, pred_cols = extract_tables_and_columns(pred_str)

        # Table metrics: precision (ratio of correct predictions), recall (ratio of ground truth covered)
        table_tp = len(truth_tables & pred_tables)
        table_fp = len(pred_tables - truth_tables)
        table_fn = len(truth_tables - pred_tables)

        table_precision = table_tp / (table_tp + table_fp) if (table_tp + table_fp) else 1.0
        table_recall = table_tp / (table_tp + table_fn) if (table_tp + table_fn) else 1.0

        # Column metrics: same as above
        col_tp = len(truth_cols & pred_cols)
        col_fp = len(pred_cols - truth_cols)
        col_fn = len(truth_cols - pred_cols)

        col_precision = col_tp / (col_tp + col_fp) if (col_tp + col_fp) else 1.0
        col_recall = col_tp / (col_tp + col_fn) if (col_tp + col_fn) else 1.0

        return table_precision, table_recall, col_precision, col_recall

    def evaluate_results(self, results: List[Dict]) -> Dict:
        """
        Batch evaluate results: calculate various metrics for tables and columns
        :param results: List of processing results (containing ground truth and predictions)
        :return: Evaluation metrics dictionary (including average accuracy, precision, recall)
        """
        total_samples = len(results)
        if total_samples == 0:
            return {"error": "No evaluation data"}

        # Initialize metrics containers
        metrics = {
            "table": {"partial_acc": [], "strict_acc": [], "precision": [], "recall": []},
            "column": {"partial_acc": [], "strict_acc": [], "precision": [], "recall": []}
        }

        for res in results:
            gt_str = res["ground_truth_standard_str"]
            pred_str = res["pred_standard_str"]

            # 1. Calculate partial/strict accuracy
            table_partial, col_partial = self.calculate_partial_accuracy(gt_str, pred_str)
            table_strict, col_strict = self.calculate_strict_accuracy(gt_str, pred_str)
            metrics["table"]["partial_acc"].append(table_partial)
            metrics["column"]["partial_acc"].append(col_partial)
            metrics["table"]["strict_acc"].append(table_strict)
            metrics["column"]["strict_acc"].append(col_strict)

            # 2. Calculate precision and recall
            table_precision, table_recall, col_precision, col_recall = self.calculate_precision_recall(gt_str, pred_str)
            metrics["table"]["precision"].append(table_precision)
            metrics["table"]["recall"].append(table_recall)
            metrics["column"]["precision"].append(col_precision)
            metrics["column"]["recall"].append(col_recall)

        # Calculate averages
        for obj in ["table", "column"]:
            metrics[obj]["avg_partial_acc"] = np.mean(metrics[obj]["partial_acc"])
            metrics[obj]["avg_strict_acc"] = np.mean(metrics[obj]["strict_acc"])
            metrics[obj]["avg_precision"] = np.mean(metrics[obj]["precision"])
            metrics[obj]["avg_recall"] = np.mean(metrics[obj]["recall"])

        return metrics