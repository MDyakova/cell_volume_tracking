import argparse
import glob
import logging
import os
import sys
import inspect
import pylint.lint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=os.getcwd(), help="Path to lint")
    parser.add_argument(
        "--threshold", type=float, default=7.0, help="Minimum acceptable score"
    )
    args = parser.parse_args()

    path = os.path.abspath(args.path)
    logger.info("PyLint Starting | Path: %s | Threshold: %.1f", path, args.threshold)

    # Collect all .py files under path (recursive)
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "**", "*.py"), recursive=True)
        if not files:
            logger.warning("No Python files found under %s", path)
            sys.exit(0)
    else:
        files = [path]

    # Common flags
    pylint_args = [
        "--disable=E1136,E1137",
        # Add other config here if you like, e.g. "--errors-only"
    ] + files

    # Run pylint, handling old/new keyword
    sig = inspect.signature(pylint.lint.Run)
    if "do_exit" in sig.parameters:
        results = pylint.lint.Run(pylint_args, do_exit=False)  # Pylint < 3.0
    else:
        results = pylint.lint.Run(pylint_args, exit=False)  # Pylint ≥ 3.0

    # Extract global score across Pylint versions
    score = None
    stats = results.linter.stats
    try:
        # Pylint ≥ 3.0 (dataclass with attributes)
        score = getattr(stats, "global_note")
    except Exception:
        pass
    if score is None:
        try:
            # Pylint 2.x (dict-like)
            score = stats["global_note"]
        except Exception:
            pass

    if score is None:
        logger.error("Could not read Pylint score (global_note).")
        sys.exit(1)

    print(f"PyLint score: {score:.2f} / 10")
    if float(score) < float(args.threshold):
        logger.error("Score %.2f is below threshold %.2f", score, args.threshold)
        sys.exit(1)
    else:
        logger.info("Score %.2f meets threshold %.2f", score, args.threshold)
        sys.exit(0)


if __name__ == "__main__":
    main()
