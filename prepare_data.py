"""保留时间轴的检查、时间划分、有效窗口和训练期标准化。"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shlex
import sys
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
FEATURES = [
    "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)",
    "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)",
    "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)",
    "wd (deg)", "rain (mm)", "raining (s)", "SWDR (W/m²)",
    "PAR (µmol/m²/s)", "max. PAR (µmol/m²/s)", "Tlog (degC)", "CO2 (ppm)",
]
LENGTH = 12
REASONS = ["insufficient_history", "time_gap", "invalid_input", "invalid_target"]


class DataError(ValueError):
    def __init__(self, message, **details):
        super().__init__(message)
        self.details = details


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_checked(path, output):
    """不排序、不删除无效行；异常含原始零起始行索引。"""
    # csv.reader 先验证宽度，避免 pandas 自动解释多余列或重命名重复表头。
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header != ["date", *FEATURES]:
            raise DataError("CSV 字段或顺序不符合约定", expected=["date", *FEATURES], actual=header)
        for i, row in enumerate(reader):
            if len(row) != len(header):
                raise DataError("CSV 记录列数错误", original_row=i, actual_columns=len(row))
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, skip_blank_lines=False)
    if raw.empty:
        raise DataError("CSV 没有观测记录")
    dates = pd.to_datetime(raw["date"], format="%Y/%m/%d %H:%M", errors="coerce")
    bad_dates = np.flatnonzero(dates.isna().to_numpy()).tolist()
    if bad_dates:
        raise DataError("时间戳无法解析", original_rows=bad_dates,
                        values=raw.loc[bad_dates, "date"].tolist())
    text = raw[FEATURES]
    numeric = text.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    blank = text.apply(lambda col: col.str.strip().eq("")).to_numpy()
    # 文字 nan/inf 归为非有限数，其他不可转换文本归为非数值。
    explicit_nan = text.apply(lambda col: col.str.strip().str.lower().isin(
        ["nan", "+nan", "-nan"])).to_numpy()
    masks = {
        "empty": blank,
        "non_numeric": np.isnan(numeric) & ~blank & ~explicit_nan,
        "non_finite": np.isinf(numeric) | explicit_nan,
        "sentinel_minus_9999": numeric == -9999,
    }
    invalid = ~np.isfinite(numeric) | (numeric == -9999)
    invalid_cells = []
    for i, j in np.argwhere(invalid):
        invalid_cells.append({
            "original_row": int(i), "timestamp": dates.iloc[i].isoformat(),
            "feature": FEATURES[j], "raw_value": text.iloc[i, j],
            "reasons": "|".join(name for name, mask in masks.items() if mask[i, j]),
        })
    pd.DataFrame(invalid_cells, columns=["original_row", "timestamp", "feature", "raw_value", "reasons"]).to_csv(
        output / "invalid_cells.csv", index=False)
    checks = {
        "original_rows": len(raw), "feature_order": FEATURES,
        "time_parse_errors": [], "numeric_dtypes_after_conversion": [str(numeric.dtype)] * len(FEATURES),
        "invalid_policy": "非有限数、空白、不可转换数值和 -9999 无效；不填补、不删除时间轴观测",
        "invalid_cells_original": int(invalid.sum()),
        "invalid_rows_original": int(invalid.any(axis=1).sum()),
        "invalid_by_feature_original": dict(zip(FEATURES, invalid.sum(axis=0).tolist())),
        "invalid_by_reason_original": {name: int(mask.sum()) for name, mask in masks.items()},
    }
    save_json(output / "data_checks.json", checks)
    keep, duplicates = [], []
    for i in range(len(raw)):
        if i and dates.iloc[i] < dates.iloc[i - 1]:
            raise DataError("原始时间倒退，禁止静默排序", original_row=i, previous_original_row=i - 1,
                            timestamp=dates.iloc[i].isoformat(), previous_timestamp=dates.iloc[i - 1].isoformat())
        if keep and dates.iloc[i] == dates.iloc[keep[-1]]:
            first = keep[-1]
            # 有效数值按数值比较；不可转换单元格还必须原文字一致，避免 NaN 掩盖冲突。
            same = (numeric[i] == numeric[first]) | (
                np.isnan(numeric[i]) & np.isnan(numeric[first])
                & (text.iloc[i].to_numpy() == text.iloc[first].to_numpy()))
            if not same.all():
                raise DataError("同一时间戳指标冲突", original_row=i, retained_original_row=first,
                                timestamp=dates.iloc[i].isoformat(),
                                conflicting_features=np.array(FEATURES)[~same].tolist())
            duplicates.append({"deleted_original_row": i, "retained_original_row": first,
                               "timestamp": dates.iloc[i].isoformat()})
        else:
            keep.append(i)
    pd.DataFrame(duplicates, columns=["deleted_original_row", "retained_original_row", "timestamp"]).to_csv(
        output / "duplicate_mapping.csv", index=False)
    stamps = dates.to_numpy(dtype="datetime64[ns]")
    original_intervals = np.diff(stamps).astype("timedelta64[s]").astype(np.int64)
    rows = np.asarray(keep, dtype=np.int64)
    stamps, numeric, invalid = stamps[rows], numeric[rows], invalid[rows]
    intervals = np.diff(stamps).astype("timedelta64[s]").astype(np.int64)
    checks.update({
        "deduplicated_rows": len(rows), "duplicates_removed": len(duplicates),
        "original_interval_counts_seconds": dict(Counter(map(str, original_intervals))),
        "deduplicated_interval_counts_seconds": dict(Counter(map(str, intervals))),
        "invalid_cells_deduplicated": int(invalid.sum()),
        "invalid_rows_deduplicated": int(invalid.any(axis=1).sum()),
        "irregular_intervals": [{"before_original_row": int(rows[j]),
                                 "after_original_row": int(rows[j + 1]),
                                 "before_timestamp": str(stamps[j]), "after_timestamp": str(stamps[j + 1]),
                                 "seconds": int(intervals[j])}
                                for j in np.flatnonzero(intervals != 600)],
    })
    save_json(output / "data_checks.json", checks)
    return rows, stamps, numeric, invalid, checks


def fixed_splits(n):
    train_end, val_count = n * 70 // 100, n * 15 // 100
    return {"train": (0, train_end), "val": (train_end, train_end + val_count),
            "test": (train_end + val_count, n)}


def build_windows(rows, stamps, numeric, invalid, splits):
    """返回有效目标位置和覆盖全部去重位置的窗口审计表。"""
    targets = {name: [] for name in splits}
    records = []
    row_invalid = invalid.any(axis=1)
    for name, (lo, hi) in splits.items():
        for j in range(lo, hi):
            start = max(0, j - LENGTH)
            flags = {
                "insufficient_history": j < LENGTH,
                "time_gap": bool(np.any(np.diff(stamps[start:j + 1]) != np.timedelta64(10, "m"))),
                "invalid_input": bool(row_invalid[start:j].any()),
                "invalid_target": bool(invalid[j, -1]),
            }
            valid = not any(flags.values())
            if valid:
                targets[name].append(j)
            record = {"split": name, "target_position": j, "target_original_row": int(rows[j]),
                      "target_timestamp": str(stamps[j]), "valid": valid, **flags,
                      "exclusion_reasons": "|".join(key for key, value in flags.items() if value)}
            # 不足历史时只记录实际可用行，不构造虚假输入。
            for k in range(LENGTH):
                pos = j - LENGTH + k
                record[f"input_{k}_original_row"] = int(rows[pos]) if pos >= 0 else None
                record[f"input_{k}_timestamp"] = str(stamps[pos]) if pos >= 0 else None
            records.append(record)
    audit = pd.DataFrame(records)
    for k in range(LENGTH):
        audit[f"input_{k}_original_row"] = audit[f"input_{k}_original_row"].astype("Int64")
    return {key: np.asarray(value, dtype=np.int64) for key, value in targets.items()}, audit


def standardize(numeric, invalid, train_end):
    fit_positions = np.flatnonzero(~invalid[:train_end].any(axis=1))
    if len(fit_positions) < 2:
        raise DataError("训练时间段至少需要两条全部指标有效的观测", valid_training_rows=len(fit_positions))
    scaler = StandardScaler().fit(numeric[fit_positions])
    constant = np.all(numeric[fit_positions] == numeric[fit_positions[0]], axis=0)
    return scaler, fit_positions, np.array(FEATURES)[constant].tolist()


def make_arrays(rows, stamps, numeric, targets, scaler):
    arrays = {"feature_names": np.asarray(FEATURES), "window_length": np.asarray(LENGTH)}
    for name, positions in targets.items():
        if not len(positions):
            raise DataError("数据段没有有效窗口", split=name)
        inputs = positions[:, None] + np.arange(-LENGTH, 0)
        raw_x = numeric[inputs]
        x = scaler.transform(raw_x.reshape(-1, len(FEATURES))).reshape(raw_x.shape)
        y = numeric[positions, -1].copy()
        last_co2 = numeric[positions - 1, -1].copy()
        if x.shape != (len(positions), LENGTH, len(FEATURES)) or y.shape != (len(positions),):
            raise DataError("窗口或标签形状错误", split=name)
        if not all(np.isfinite(a).all() for a in (x, y, last_co2)):
            raise DataError("标准化后出现非有限值", split=name)
        if not np.all(np.diff(stamps[positions]) > np.timedelta64(0, "ns")):
            raise DataError("目标时间未严格递增", split=name)
        arrays.update({f"X_{name}": x, f"y_{name}": y, f"last_co2_{name}": last_co2,
                       f"target_original_rows_{name}": rows[positions],
                       f"target_timestamps_{name}": stamps[positions]})
    return arrays


def run(input_path, output):
    """output 必须尚不存在；失败产物保留且不能直接重用目录。"""
    input_path, output = Path(input_path).resolve(), Path(output).resolve()
    forbidden = [ROOT / "dataset", input_path.parent]
    if any(output == folder or folder in output.parents for folder in forbidden):
        raise DataError("输出不能位于原始数据目录", output_dir=str(output))
    output.mkdir(parents=True, exist_ok=False)
    stage = "provenance"
    started = time.perf_counter()
    try:
        config = {
            "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": str(input_path), "output_dir": str(output),
            "command": shlex.join([sys.executable, *sys.argv]),
            "working_directory": str(Path.cwd()), "data_sha256": sha256(input_path),
            "code_sha256": {name: sha256(ROOT / name) for name in
                            ["prepare_data.py", "requirements.txt", "test_prepare_data.py"]},
            "window_length": LENGTH, "feature_order": FEATURES, "target_unit": "ppm",
            "split_ratios": [0.70, 0.15, 0.15], "seed_convention": 42,
            "randomness_used": False, "invalid_sentinel": -9999,
        }
        save_json(output / "config.json", config)
        save_json(output / "environment.json", {
            "python": sys.version, "platform": platform.platform(),
            "packages": {name: version(name) for name in
                         ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl",
                          "cloudpickle", "narwhals", "python-dateutil", "six", "tzdata"]},
        })
        stage = "load_and_check"
        rows, stamps, numeric, invalid, checks = load_checked(input_path, output)
        stage = "split_and_window"
        splits = fixed_splits(len(rows))
        targets, windows = build_windows(rows, stamps, numeric, invalid, splits)
        windows.to_csv(output / "window_indices.csv", index=False)
        boundaries = {}
        for name, (lo, hi) in splits.items():
            if hi <= lo:
                raise DataError("时间段为空", split=name)
            boundaries[name] = {
                "start_position": lo, "stop_position_exclusive": hi, "row_count": hi - lo,
                "start_original_row": int(rows[lo]), "end_original_row": int(rows[hi - 1]),
                "start_timestamp": str(stamps[lo]), "end_timestamp": str(stamps[hi - 1]),
                "valid_target_positions": targets[name].tolist(),
                "valid_target_original_rows": rows[targets[name]].tolist(),
            }
        save_json(output / "split_indices.json", boundaries)
        stats = {}
        for name in splits:
            part = windows[windows["split"] == name]
            complete = part[~part["insufficient_history"]]
            stats[name] = {
                "target_positions": len(part), "complete_candidate_windows": len(complete),
                "valid_windows": int(part["valid"].sum()),
                "excluded_target_positions": int((~part["valid"]).sum()),
                "excluded_complete_windows": int((~complete["valid"]).sum()),
                "reason_counts_all_positions": {r: int(part[r].sum()) for r in REASONS},
                "reason_counts_complete_windows": {r: int(complete[r].sum()) for r in REASONS},
            }
        checks["windows"] = stats
        save_json(output / "data_checks.json", checks)
        stage = "standardize"
        scaler, fit_positions, constants = standardize(numeric, invalid, splits["train"][1])
        pd.DataFrame({"position": fit_positions, "original_row": rows[fit_positions],
                      "timestamp": stamps[fit_positions]}).to_csv(output / "scaler_fit_rows.csv", index=False)
        save_json(output / "scaler_stats.json", {
            "fit_row_count": len(fit_positions), "feature_order": FEATURES,
            "mean": scaler.mean_.tolist(), "variance_ddof_0": scaler.var_.tolist(),
            "scale": scaler.scale_.tolist(), "constant_columns": constants,
        })
        checks["constant_training_columns"] = constants
        save_json(output / "data_checks.json", checks)
        stage = "save_arrays"
        arrays = make_arrays(rows, stamps, numeric, targets, scaler)
        np.savez_compressed(output / "dataset.npz", **arrays)
        joblib.dump(scaler, output / "scaler.joblib")
        stage = "verify_saved_artifacts"
        with np.load(output / "dataset.npz", allow_pickle=False) as saved:
            for key, value in arrays.items():
                if not np.array_equal(saved[key], value):
                    raise DataError("数组保存后不一致", key=key)
        restored = joblib.load(output / "scaler.joblib")
        for attribute in ["mean_", "var_", "scale_", "n_samples_seen_"]:
            if not np.array_equal(getattr(restored, attribute), getattr(scaler, attribute)):
                raise DataError("缩放器保存不一致", attribute=attribute)
        if sha256(input_path) != config["data_sha256"]:
            raise DataError("原始数据哈希发生变化")
        config["status"] = "complete"
        save_json(output / "config.json", config)
        summary = {"status": "complete", "deduplicated_rows": len(rows), "splits": stats,
                   "scaler_fit_rows": len(fit_positions), "constant_columns": constants,
                   "original_data_unchanged": True, "saved_arrays_verified": True,
                   "elapsed_seconds": time.perf_counter() - started}
        save_json(output / "summary.json", summary)
        return summary
    except Exception as exc:
        if (output / "config.json").exists():
            config["status"] = "failed"
            save_json(output / "config.json", config)
        save_json(output / "failure.json", {
            "stage": stage, "exception": type(exc).__name__, "message": str(exc),
            "details": getattr(exc, "details", {}), "traceback": traceback.format_exc(),
        })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "dataset" / "weather.csv",
                        help="原始 CSV；显式相对路径按当前工作目录解析")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / datetime.now(timezone.utc).strftime("prepare_%Y%m%dT%H%M%S_%fZ"),
                        help="新建输出目录，禁止覆盖已有目录或写入原始数据目录")
    args = parser.parse_args()
    try:
        summary = run(args.input, args.output_dir)
    except Exception as exc:
        print(f"预处理失败：{exc}", file=sys.stderr)
        return 1
    print(f"预处理完成：{args.output_dir.resolve()}")
    print(json.dumps({name: part["valid_windows"] for name, part in summary["splits"].items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
