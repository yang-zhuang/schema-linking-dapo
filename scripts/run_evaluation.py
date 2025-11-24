#!/usr/bin/env python3
"""
Run Qwen3-0.6B Model Evaluation
Convenient script to run the evaluation
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import main

if __name__ == "__main__":
    main()