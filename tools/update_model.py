"""Install a new model file into the project.

Usage (from the project folder):
    python tools/update_model.py path/to/new_model.joblib
    python tools/update_model.py path/to/new_model.joblib --keep-active

By default the new model becomes models/final_model.joblib and the previous one is archived in
models/archive/. With --keep-active it is only copied into models/ under its own name, and you
select it in the app sidebar.

SECURITY: a .joblib file is a pickle and can run code when it is loaded. Only install files you created yourself.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src import modeling


def main():
    parser = argparse.ArgumentParser(description="Install a new model bundle.")
    parser.add_argument("path", help="path to the new .joblib model file")
    parser.add_argument("--keep-active", action="store_true", help="do not replace the active model")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        sys.exit(f"File not found: {path}")

    try:
        if args.keep_active:
            result = modeling.install_bundle(path, models_dir=ROOT / "models", activate=False, name=path.name)
        else:
            result = modeling.install_bundle(path, models_dir=ROOT / "models")
    except ValueError as err:
        sys.exit(f"Not installed, the model file has a problem: {err}")

    print(f"installed: {result['installed']}")
    if result["archived"]:
        print(f"previous model archived as: {result['archived']}")


if __name__ == "__main__":
    main()
