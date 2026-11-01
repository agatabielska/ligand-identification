"""Command-line options shared by ensemble experiment runners."""

import argparse
import sys


def parse_experiment_args() -> argparse.Namespace:
    """Parse and remove the custom option before Hydra processes argv."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--experiment-number")
    parser.add_argument("--run-id")
    args, remaining = parser.parse_known_args()
    sys.argv[1:] = remaining
    return args
