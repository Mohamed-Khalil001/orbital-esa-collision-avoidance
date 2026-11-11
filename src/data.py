"""Loading the data and checking its shape."""
import pandas as pd

from . import config


def resolve_train_path(data_dir=None):
    data_dir = data_dir or config.DATA_DIR
    for name in ("train_data.zip", "train_data.csv"):
        path = data_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"train_data.zip (or .csv) not found in {data_dir}")


def check_schema(df, need_risk=True):
    required = [config.ID_COL, config.TIME_COL] + ([config.RISK_COL] if need_risk else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def load_train(path=None):
    df = pd.read_csv(path or resolve_train_path())
    return check_schema(df, need_risk=True)


def load_test(path=None):
    df = pd.read_csv(path or (config.DATA_DIR / "test_data.csv"))
    return check_schema(df, need_risk=False)


def load_demo_sample(path=None):
    """Loads the bundled 150-event demo sample from data/sample/demo_events.csv."""
    sample_path = path or (config.SAMPLE_DIR / "demo_events.csv")
    df = pd.read_csv(sample_path)
    return check_schema(df, need_risk=True)


def load_sample(path=None):
    """Alias for load_demo_sample."""
    return load_demo_sample(path=path)
