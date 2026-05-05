"""
Customer Churn Prediction — Main Pipeline
==========================================
Run this file to execute the full pipeline end-to-end:
    1. Exploratory Data Analysis  (EDA)
    2. Model Training & Evaluation
    3. Sample Prediction

Usage:
    python main.py            # full pipeline
    python main.py --eda      # EDA only
    python main.py --train    # training only
    python main.py --predict  # prediction demo (model must already be trained)
"""

import sys
import os
import argparse

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def run_eda():
    print("\n" + "█"*60)
    print("  STEP 1 — Exploratory Data Analysis")
    print("█"*60)
    from src.eda import run_eda as _run_eda
    _run_eda()


def run_training():
    print("\n" + "█"*60)
    print("  STEP 2 — Model Training & Evaluation")
    print("█"*60)
    from src.train import run_training as _run_training
    _run_training()


def run_prediction_demo():
    print("\n" + "█"*60)
    print("  STEP 3 — Prediction Demo")
    print("█"*60)
    # Run the predict script's __main__ block
    import runpy
    runpy.run_path(
        os.path.join(os.path.dirname(__file__), "src", "predict.py"),
        run_name="__main__"
    )


def main():
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Pipeline")
    parser.add_argument("--eda",     action="store_true", help="Run EDA only")
    parser.add_argument("--train",   action="store_true", help="Run training only")
    parser.add_argument("--predict", action="store_true", help="Run prediction demo only")
    args = parser.parse_args()

    if args.eda:
        run_eda()
    elif args.train:
        run_training()
    elif args.predict:
        run_prediction_demo()
    else:
        # Default: full pipeline
        run_eda()
        run_training()
        run_prediction_demo()
        print("\n" + "█"*60)
        print("  ALL DONE! Check plots/ and reports/ for outputs.")
        print("█"*60 + "\n")


if __name__ == "__main__":
    main()
