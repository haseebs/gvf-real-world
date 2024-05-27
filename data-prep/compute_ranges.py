"""
This script generate the `ranges.json` file for PROD mode
"""
import json
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from multiprocessing import Pool
from functools import reduce
import numpy as np


def get_mode(df):
    return df["ZW_CTRL_P.STEP"] // 100


def drop_useless_columns(df):
    keywords = ["MC", "BW", "MIT", "RC", "DRAIN", "ZZZZZZ", "timestamp"]
    for kw in keywords:
        df = df.drop(columns=[c for c in df.columns if kw in c])
    return df


def merge_ranges(a, b):
    if not a:
        return b

    if not b:
        return a

    common = set(a.keys()) & set(b.keys())
    out = {}
    for k in common:
        out[k] = {}
        out[k]["min"] = min(a[k]["min"], b[k]["min"])
        out[k]["max"] = max(a[k]["max"], b[k]["max"])
    return out


def get_percentile_normalization_ranges(x):
    try:
        p_high = np.percentile(x, 95)  # 95th percentile.
        p_low = np.percentile(x, 5)
        min_p = max(min(x), p_low)
        max_p = min(max(x), p_high)
        return pd.Series(index=['min', 'max'], data=[min_p, max_p])
    except:
        # for non-numerical data
        return pd.Series(index=['min', 'max'], data=[0, 0])


def get_min_max(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"{csv_path} is empty")
        return {}
    except Exception as e:
        print(f"{csv_path} cannot be read due to {e}")
        return {}

    df.columns = [c.split("Program:")[-1].replace("ZW.", "ZW_") for c in df.columns]
    df = drop_useless_columns(df)
    modes = get_mode(df)
    df = df[modes == 13]
    df = df.replace({False: 0, True: 1})
    get_percentile_normalization_ranges(df)
    current_ranges = df.apply(get_percentile_normalization_ranges).to_dict()
    return current_ranges


if __name__ == "__main__":
    # DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "logs"
    DATA_DIR = Path(__file__).resolve().parents[1] / "data-logs/logs/"
    print(f"data dir: {DATA_DIR}")

    with Pool(11) as p:
        ranges = p.map(get_min_max, DATA_DIR.glob("*.log.gz"))

    numericals = reduce(merge_ranges, ranges)

    # delete categorical columns
    del numericals["ZW_PROD_SP.KY1"]
    del numericals["ZW_CTRL_P.STEP"]

    # delete constant values
    numericals = {k: v for k, v in numericals.items() if v["min"] != v["max"]}

    categories = {
        "ZW_CTRL.MODE": [1, 2, 3, 4, 5, 13, 31, 32, 33, 45, 61, 83],
        "ZW_PROD_SP.KY1": [0, 10, 20],
    }

    ranges_json = {
        "categorical": categories,
        "numerical": {k: [v["min"], v["max"]] for k, v in numericals.items()},
        "request_tags": list(numericals.keys()) + ["ZW_PROD_SP.KY1", "ZW_CTRL_P.STEP"],
    }

    with open(Path(__file__).parents[1] / "data-logs/configs" / "4m_ranges.json", "w") as f:
        json.dump(ranges_json, f)
