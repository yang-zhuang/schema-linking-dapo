"""
Refactored evaluation script for Qwen3-0.6B model
Modular design with English naming
"""

import os
import time
import argparse
from tqdm import tqdm
from typing import Dict

from .config import (
    DEFAULT_API_URL, DEFAULT_API_KEY, DEFAULT_MODEL_NAME,
    DEFAULT_INPUT_CSV_PATH, DEFAULT_SAVE_PATH, DEFAULT_METRICS_PATH,
    DEFAULT_SYSTEM_PROMPT, DEFAULT_MAX_WORKERS, DEFAULT_REQUEST_TIMEOUT
)
from .data_loader import DataLoader
from .model_client import ModelClient
from .metrics_calculator import MetricsCalculator
from .file_handler import FileHandler


class EvaluationRunner:
    """Main evaluation runner that orchestrates the evaluation process"""

    def __init__(self, args):
        self.args = args
        self.data_loader = DataLoader()
        self.model_client = ModelClient(
            api_url=args.api_url,
            api_key=args.api_key,
            model_name=args.model_name,
            system_prompt=args.system_prompt
        )
        self.metrics_calculator = MetricsCalculator()
        self.file_handler = FileHandler()

    def run(self):
        """Main evaluation workflow"""
        # 1. Initialize components
        print("Initializing components...")
        client = self.model_client.init_client()
        df = self.data_loader.load_input_data(self.args.input_csv)

        # 2. Prepare data
        prompts, ground_truth_data = self.data_loader.prepare_data(
            df, debug=self.args.debug, debug_samples=self.args.debug_samples
        )

        # 3. Batch concurrent model calls
        print(f"\nStarting concurrent processing of {len(prompts)} samples, workers: {self.args.max_workers}")
        start_time = time.time()

        model_raw_responses = self.model_client.call_model_batch(
            prompts, max_workers=self.args.max_workers
        )

        end_time = time.time()
        print(f"Concurrent processing completed, total time: {end_time - start_time:.2f} seconds")

        # 4. Parse results
        print("Parsing model responses...")
        results = []
        for i, (model_raw_resp, gt_data) in tqdm(
                enumerate(zip(model_raw_responses, ground_truth_data)),
                total=len(model_raw_responses),
                desc="Result parsing"
        ):
            # Parse prediction and standardize
            pred_json = self.model_client.parse_model_response(model_raw_resp)
            pred_str = self.model_client.convert_pred_to_standard_str(pred_json)

            # Store single result
            results.append({
                "question": gt_data["question"],
                "model_raw_response": model_raw_resp,
                "pred_json": pred_json,
                "pred_standard_str": pred_str,
                "ground_truth_json": gt_data["ground_truth_json"],
                "ground_truth_standard_str": gt_data["ground_truth_standard_str"]
            })

        # 5. Evaluate and print results
        eval_metrics = self.metrics_calculator.evaluate_results(results)
        self._print_evaluation_summary(eval_metrics)

        # 6. Save results and metrics
        self.file_handler.save_process_results(results, self.args.save_path)
        self.file_handler.save_evaluation_metrics(eval_metrics, self.args.metrics_path)

        return eval_metrics

    def _print_evaluation_summary(self, eval_metrics: Dict):
        """Print evaluation results summary"""
        print("\n" + "=" * 50)
        print("Evaluation Results Summary")
        print("=" * 50)

        # Print table metrics
        print("\n【Table Metrics】")
        print(f"Average Partial Accuracy: {eval_metrics['table']['avg_partial_acc']:.4f}")
        print(f"Average Strict Accuracy: {eval_metrics['table']['avg_strict_acc']:.4f}")
        print(f"Average Precision: {eval_metrics['table']['avg_precision']:.4f}")
        print(f"Average Recall: {eval_metrics['table']['avg_recall']:.4f}")

        # Print column metrics
        print("\n【Column Metrics】")
        print(f"Average Partial Accuracy: {eval_metrics['column']['avg_partial_acc']:.4f}")
        print(f"Average Strict Accuracy: {eval_metrics['column']['avg_strict_acc']:.4f}")
        print(f"Average Precision: {eval_metrics['column']['avg_precision']:.4f}")
        print(f"Average Recall: {eval_metrics['column']['avg_recall']:.4f}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Schema linking assistant evaluation script")

    # vLLM service configuration
    parser.add_argument("--api_url", type=str, default=DEFAULT_API_URL,
                        help="vLLM API service address")
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY,
                        help="API key (vLLM doesn't require authentication, only for SDK format)")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME,
                        help="Model name")

    # Data path configuration
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT_CSV_PATH,
                        help="Input CSV file path")
    parser.add_argument("--save_path", type=str, default=DEFAULT_SAVE_PATH,
                        help="Result save path")
    parser.add_argument("--metrics_path", type=str, default=DEFAULT_METRICS_PATH,
                        help="Evaluation metrics save path")

    # Concurrency configuration
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help="Number of concurrent worker threads")
    parser.add_argument("--request_timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT,
                        help="Single request timeout (seconds)")

    # System prompt
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
                        help="System prompt")
    parser.add_argument("--system_prompt_file", type=str, default=None,
                        help="System prompt file path")

    # Debug mode
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (process only a few samples)")
    parser.add_argument("--debug_samples", type=int, default=5,
                        help="Number of samples to process in debug mode")

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()

    # Load system prompt from file if provided
    if args.system_prompt_file and os.path.exists(args.system_prompt_file):
        try:
            with open(args.system_prompt_file, 'r', encoding='utf-8') as f:
                args.system_prompt = f.read().strip()
            print(f"System prompt loaded from file: {args.system_prompt_file}")
        except Exception as e:
            print(f"Failed to read system prompt file: {e}, using default prompt")

    print("=" * 60)
    print("Schema Linking Assistant Evaluation Script")
    print("=" * 60)
    print(f"Input file: {args.input_csv}")
    print(f"Model: {args.model_name}")
    print(f"API address: {args.api_url}")
    print(f"Workers: {args.max_workers}")
    print(f"Debug mode: {'Yes' if args.debug else 'No'}")
    if args.debug:
        print(f"Debug samples: {args.debug_samples}")
    print("=" * 60)

    # Run evaluation
    runner = EvaluationRunner(args)
    runner.run()


if __name__ == "__main__":
    main()