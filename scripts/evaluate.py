#!/usr/bin/env python3
"""
Model Evaluation Script for Text-to-SQL Schema Selection
Evaluates the performance of trained models on validation data
"""

import argparse
import json
import sys
import logging
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.core import (
    MetricsCalculator, DataProcessor, InferenceManager,
    InferenceConfig, ReportGenerator
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate Text-to-SQL schema selection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model"
    )
    parser.add_argument(
        "--data-source",
        type=str,
        required=True,
        help="Validation data source file (CSV or JSONL)"
    )

    # Model configuration
    parser.add_argument(
        "--model-type",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="Model inference type"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature"
    )

    # LoRA configuration
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA weights"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        help="Path to LoRA weights"
    )

    # Data configuration
    parser.add_argument(
        "--source-type",
        type=str,
        default="csv",
        choices=["csv", "jsonl"],
        help="Data source type"
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        help="Limit number of samples for evaluation"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="model_evaluation",
        help="Base name for evaluation reports"
    )

    # Other options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--no-console-summary",
        action="store_true",
        help="Skip console summary output"
    )

    return parser.parse_args()


def create_inference_config(args) -> InferenceConfig:
    """Create inference configuration from arguments"""
    return InferenceConfig(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_lora=args.use_lora,
        lora_path=args.lora_path,
        vllm_enabled=(args.model_type == "vllm")
    )


def main():
    """Main evaluation function"""
    args = parse_arguments()
    setup_logging(args.verbose)

    print("="*60)
    print("🚀 Starting Model Evaluation")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_source}")
    print(f"Output: {args.output_dir}")
    if args.sample_limit:
        print(f"Samples: {args.sample_limit}")
    print("-"*60)

    try:
        # Initialize components
        print("📦 Initializing evaluation components...")

        # Data processor
        data_processor = DataProcessor()

        # Load evaluation data
        print(f"📊 Loading evaluation data from {args.data_source}...")
        if args.source_type == "csv":
            samples = data_processor.load_from_csv(args.data_source, has_ground_truth=True)
        else:
            samples = data_processor.load_from_jsonl(args.data_source)

        # Filter samples
        samples = data_processor.filter_samples(samples, sample_limit=args.sample_limit)
        print(f"✅ Loaded {len(samples)} samples")

        # Create prompts
        prompts = data_processor.create_prompts(samples)

        # Initialize inference manager
        print("🔧 Initializing inference engine...")
        inference_config = create_inference_config(args)
        inference_manager = InferenceManager(inference_config)

        # Initialize metrics calculator and report generator
        metrics_calculator = MetricsCalculator()
        report_generator = ReportGenerator(args.output_dir)

        # Run inference
        print(f"🚀 Running inference on {len(prompts)} samples...")
        inference_results = inference_manager.run_inference(prompts)

        # Calculate metrics
        print("📈 Calculating evaluation metrics...")
        metrics_list = []

        for i, (sample, inference_result, prompt) in enumerate(zip(samples, inference_results, prompts)):
            try:
                inference_time = inference_result.get('inference_time', 0.0)
                generated_text = inference_result.get('generated_text', '')

                # Calculate metrics for this sample
                metrics = metrics_calculator.calculate_single_metrics(
                    prediction=generated_text,
                    ground_truth=sample.ground_truth,
                    inference_time=inference_time
                )
                metrics_list.append(metrics)

            except Exception as e:
                logging.error(f"Error processing sample {i}: {e}")
                # Create empty metrics for failed samples
                empty_metrics = metrics_calculator.calculate_single_metrics(
                    prediction="",
                    ground_truth={'selected_tables': [], 'selected_columns': []},
                    inference_time=0.0
                )
                metrics_list.append(empty_metrics)

        # Aggregate metrics
        print("📊 Aggregating results...")
        aggregated_metrics = metrics_calculator.aggregate_metrics(metrics_list)

        # Generate reports
        print("📄 Generating evaluation reports...")

        # JSON report
        json_report = report_generator.generate_json_report(
            metrics_list=metrics_list,
            aggregated_metrics=aggregated_metrics,
            samples=samples,
            report_name=args.report_name
        )

        # CSV report
        csv_report = report_generator.generate_csv_report(
            metrics_list=metrics_list,
            samples=samples,
            report_name=args.report_name
        )

        # Summary report
        summary = report_generator.generate_summary_report(
            aggregated_metrics=aggregated_metrics,
            model_path=args.model_path,
            data_source=args.data_source,
            report_name=f"{args.report_name}_summary"
        )

        # Print console summary
        if not args.no_console_summary:
            report_generator.print_console_summary(aggregated_metrics)

        # Cleanup
        print("🧹 Cleaning up resources...")
        inference_manager.cleanup()

        print("\n🎉 Evaluation completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"📊 JSON Report: {json_report}")
        print(f"📈 CSV Report: {csv_report}")
        print(f"📋 Summary: {summary['performance_summary']['overall_grade']} "
              f"({summary['performance_summary']['weighted_score']:.4f})")

        return {
            "status": "success",
            "total_samples": len(samples),
            "aggregated_metrics": aggregated_metrics,
            "reports": {
                "json": json_report,
                "csv": csv_report,
                "summary": summary
            }
        }

    except KeyboardInterrupt:
        print("\n⏹️ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()