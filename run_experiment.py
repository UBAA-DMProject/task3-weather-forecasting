"""持续值与 Ridge 的固定方案训练、评价及共享实验工具。"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback

# 无显示设备也可绘图；字体及缓存不写入用户目录。
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from prepare_data import DataError, FEATURES, LENGTH, ROOT, fixed_splits, save_json, sha256


ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
METHODS = ("Persistence", "Ridge")
SNAPSHOTS = ("config.json", "environment.json", "data_checks.json", "duplicate_mapping.csv",
             "invalid_cells.csv", "split_indices.json", "window_indices.csv", "scaler_fit_rows.csv",
             "scaler_stats.json", "summary.json")


def metrics(y, prediction):
    y, prediction = np.asarray(y, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.ndim != 1 or prediction.ndim != 1 or y.shape != prediction.shape or not len(y):
        raise DataError("预测与标签必须为等长非空一维数组")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise DataError("预测或标签含非有限值")
    residual = prediction - y
    result = {"mae": float(np.mean(np.abs(residual))), "rmse": float(np.sqrt(np.mean(residual ** 2)))}
    if not all(np.isfinite(v) for v in result.values()):
        raise DataError("指标计算产生非有限值")
    return result


def improvement(baseline, value):
    return (baseline - value) / baseline * 100.0 if baseline > 0 else None


def flatten(x):
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1:] != (LENGTH, len(FEATURES)) or not np.isfinite(x).all():
        raise DataError("输入必须是有限的 M×12×21 数组")
    return x.reshape(len(x), LENGTH * len(FEATURES), order="C")


def load_prepared(directory):
    config = json.loads((directory / "config.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    splits = json.loads((directory / "split_indices.json").read_text())
    stats = json.loads((directory / "scaler_stats.json").read_text())
    if config["status"] != "complete" or summary["status"] != "complete":
        raise DataError("只接受成功完成的预处理产物")
    raw_path = Path(config["input"])
    if sha256(raw_path) != config["data_sha256"]:
        raise DataError("原始 CSV 哈希与预处理记录不符")
    if config["feature_order"] != FEATURES or config["window_length"] != LENGTH or config["target_unit"] != "ppm":
        raise DataError("预处理配置的特征、窗口或单位不符")
    if sha256(ROOT / "prepare_data.py") != config["code_sha256"]["prepare_data.py"]:
        raise DataError("预处理代码已改变，请重新运行数据准备")
    with np.load(directory / "dataset.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if data["feature_names"].tolist() != FEATURES or data["window_length"].item() != LENGTH:
        raise DataError("数组特征顺序或窗口长度不符")
    scaler = joblib.load(directory / "scaler.joblib")
    if scaler.n_features_in_ != len(FEATURES) or scaler.n_samples_seen_ != stats["fit_row_count"]:
        raise DataError("缩放器拟合记录不一致")
    for attr, key in [("mean_", "mean"), ("var_", "variance_ddof_0"), ("scale_", "scale")]:
        if not np.array_equal(getattr(scaler, attr), stats[key]):
            raise DataError("缩放器统计量不一致", attribute=attr)
    fit_rows = pd.read_csv(directory / "scaler_fit_rows.csv")
    expected_splits = fixed_splits(summary["deduplicated_rows"])
    if len(fit_rows) != stats["fit_row_count"] or not fit_rows["position"].is_unique:
        raise DataError("缩放器拟合行重复或数量不符")
    if not fit_rows["position"].between(0, expected_splits["train"][1] - 1).all():
        raise DataError("缩放器拟合行超出训练时间段")
    audit = pd.read_csv(directory / "window_indices.csv")
    previous_time, previous_row = None, None
    for split, (lo, hi) in expected_splits.items():
        boundary = splits[split]
        if (boundary["start_position"], boundary["stop_position_exclusive"]) != (lo, hi):
            raise DataError("时间划分不符合固定比例", split=split)
        x, y = data[f"X_{split}"], data[f"y_{split}"]
        flatten(x)
        metrics(y, data[f"last_co2_{split}"])
        if len(x) != len(y) or len(y) != summary["splits"][split]["valid_windows"]:
            raise DataError("窗口与标签数量不一致", split=split)
        stamps, rows = data[f"target_timestamps_{split}"], data[f"target_original_rows_{split}"]
        if stamps.shape != y.shape or rows.shape != y.shape or stamps.dtype.kind != "M" or rows.dtype.kind not in "iu":
            raise DataError("目标索引或时间戳形状／类型错误", split=split)
        if np.isnat(stamps).any() or not np.all(np.diff(stamps) > np.timedelta64(0, "ns")) or not np.all(np.diff(rows) > 0):
            raise DataError("目标时间或原始行未严格递增", split=split)
        if previous_time is not None and (stamps[0] <= previous_time or rows[0] <= previous_row):
            raise DataError("数据段目标重叠或乱序", split=split)
        previous_time, previous_row = stamps[-1], rows[-1]
        selected = audit[(audit["split"] == split) & audit["valid"]]
        if not np.array_equal(selected["target_original_row"].to_numpy(), rows):
            raise DataError("目标行与窗口审计不一致", split=split)
        if not np.array_equal(pd.to_datetime(selected["target_timestamp"]).to_numpy(dtype="datetime64[ns]"), stamps):
            raise DataError("目标时间与窗口审计不一致", split=split)
        if rows.tolist() != boundary["valid_target_original_rows"] or not selected["target_position"].between(lo, hi - 1).all():
            raise DataError("有效目标与固定边界不一致", split=split)
        # 标准化输入的历史 CO₂ 逆变换后应与持续值用的原始 ppm 一致。
        np.testing.assert_allclose(x[:, -1, -1] * scaler.scale_[-1] + scaler.mean_[-1],
                                   data[f"last_co2_{split}"], rtol=1e-12, atol=1e-10)
    return data, scaler, config, splits


def best_ridge_index(records):
    return min(range(len(records)), key=lambda i: (records[i]["rmse"], records[i]["alpha"]))


def train_ridge(x_train, y_train, x_val, y_val):
    models, records = [], []
    train, val = flatten(x_train), flatten(x_val)
    for alpha in ALPHAS:
        started = time.perf_counter()
        model = Ridge(alpha=alpha, fit_intercept=True, solver="svd").fit(train, y_train)
        fit_seconds = time.perf_counter() - started
        record = {"alpha": alpha, **metrics(y_val, model.predict(val)), "fit_seconds": fit_seconds}
        records.append(record)
        models.append(model)
        print(f"Ridge alpha={alpha:g}: validation RMSE={record['rmse']:.6f} ppm", flush=True)
    selected = best_ridge_index(records)
    for i, record in enumerate(records):
        record["selected"] = i == selected
    return models[selected], records


def gap_series(stamps, values):
    """在非连续时刻前插入 NaN，保留断点两侧的真实样本。"""
    stamps, values = np.asarray(stamps), np.asarray(values)
    gaps = np.flatnonzero(np.diff(stamps) != np.timedelta64(10, "m")) + 1
    return np.insert(stamps, gaps, stamps[gaps]), np.insert(values.astype(float), gaps, np.nan)


def save_figure(fig, directory, name):
    fig.tight_layout()
    for suffix in ["png", "svg"]:
        fig.savefig(directory / f"{name}.{suffix}", dpi=160)
    plt.close(fig)


def plot_results(data, predictions, table, test_start, directory):
    directory.mkdir()
    stamps, truth = data["target_timestamps_test"], data["y_test"]
    start = np.datetime64(test_start, "ns")
    for name, mask, title in [
        ("test_predictions", np.ones(len(stamps), dtype=bool), "One-step CO2 predictions: full test period"),
        ("test_first_24h", (stamps >= start) & (stamps < start + np.timedelta64(24, "h")),
         "One-step CO2 predictions: first 24 hours of test period"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(*gap_series(stamps[mask], truth[mask]), color="black", label="Observed", linewidth=1.1)
        for method in METHODS:
            ax.plot(*gap_series(stamps[mask], predictions[(method, "test")][mask]), label=method, alpha=0.8, linewidth=0.9)
        ax.set(title=title, ylabel="CO2 (ppm)", xlabel="Target time")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        ax.legend(ncol=4)
        ax.grid(alpha=0.2)
        save_figure(fig, directory, name)
    fig, axes = plt.subplots(len(METHODS), 1, figsize=(12, 6), sharex=True)
    for ax, method in zip(axes, METHODS):
        ax.plot(*gap_series(stamps, predictions[(method, "test")] - truth), linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set(ylabel="Residual (ppm)", title=method)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Target time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    save_figure(fig, directory, "test_residuals")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    test = table[table["split"] == "test"].set_index("model").loc[list(METHODS)]
    for ax, metric in zip(axes, ["mae", "rmse"]):
        bars = ax.bar(METHODS, test[metric])
        ax.bar_label(bars, fmt="%.3f")
        ax.set(title=f"Test {metric.upper()}", ylabel="Error (ppm)")
        ax.set_ylim(0, max(test[metric]) * 1.2)
    save_figure(fig, directory, "test_error_comparison")


def evaluate(data, predictions, test_start, methods=METHODS):
    records, rows, periods = [], [], []
    for split in ["val", "test"]:
        y = data[f"y_{split}"]
        baseline = metrics(y, predictions[("Persistence", split)])
        for method in methods:
            prediction = predictions[(method, split)]
            score = metrics(y, prediction)
            records.append({"model": method, "split": split, "n": len(y), **score,
                            **{f"{key}_improvement_pct": improvement(baseline[key], score[key]) for key in score},
                            "improvement_note": "baseline error zero: not applicable" if 0 in baseline.values() else ""})
            rows.append(pd.DataFrame({"model": method, "split": split,
                                     "target_original_row": data[f"target_original_rows_{split}"],
                                     "target_timestamp": data[f"target_timestamps_{split}"],
                                     "y_true": y, "y_pred": prediction, "residual": prediction - y}))
    stamps = data["target_timestamps_test"]
    start = np.datetime64(test_start, "ns")
    bins = ((stamps - start) // np.timedelta64(7, "D")).astype(int)
    for group in np.unique(bins):
        mask = bins == group
        for method in methods:
            periods.append({"period": int(group + 1), "period_start": str(start + group * np.timedelta64(7, "D")),
                            "period_end_exclusive": str(start + (group + 1) * np.timedelta64(7, "D")),
                            "model": method, "n": int(mask.sum()),
                            **metrics(data["y_test"][mask], predictions[(method, "test")][mask])})
    return pd.DataFrame(records), pd.concat(rows, ignore_index=True), pd.DataFrame(periods)


def run(prepared_dir, output):
    prepared_dir, output = Path(prepared_dir).resolve(), Path(output).resolve()
    if output == prepared_dir or prepared_dir in output.parents or output == ROOT / "dataset" or ROOT / "dataset" in output.parents:
        raise DataError("不能向原始数据或已有预处理目录写入实验产物")
    output.mkdir(parents=True, exist_ok=False)
    stage, config, started = "input_checks", None, time.perf_counter()
    try:
        input_hashes = {name: sha256(prepared_dir / name) for name in [*SNAPSHOTS, "dataset.npz", "scaler.joblib"]}
        data, scaler, source, boundaries = load_prepared(prepared_dir)
        config = {"status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "command": shlex.join([sys.executable, *sys.argv]), "working_directory": str(Path.cwd()),
                  "prepared_dir": str(prepared_dir), "source_data_sha256": source["data_sha256"],
                  "input_artifact_sha256": input_hashes,
                  "code_sha256": {name: sha256(ROOT / name) for name in
                                  ["run_experiment.py", "prepare_data.py", "test_experiment.py", "requirements.txt"]},
                  "methods": list(METHODS), "ridge_alphas": list(ALPHAS),
                  "seed_convention": 42, "randomness_used": False, "threads": 4, "target_unit": "ppm",
                  "test_policy": "验证选参并保存模型后才评价测试；不合并验证集重训"}
        save_json(output / "config.json", config)
        save_json(output / "environment.json", {"python": sys.version, "platform": platform.platform(),
                  "packages": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy",
                               "joblib", "matplotlib", "threadpoolctl"]}, "threads": 4})
        (output / "preprocessing").mkdir()
        for name in SNAPSHOTS:
            shutil.copyfile(prepared_dir / name, output / "preprocessing" / name)
        stage = "ridge_training"
        clock = time.perf_counter()
        with threadpool_limits(limits=4):
            ridge, search = train_ridge(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
        search_seconds = time.perf_counter() - clock
        pd.DataFrame(search).to_csv(output / "validation_search.csv", index=False)
        bundle = {"ridge": ridge, "scaler": scaler, "feature_names": FEATURES, "window_length": LENGTH,
                  "flatten_order": "C", "target_unit": "ppm"}
        joblib.dump(bundle, output / "model.joblib")
        selected = {"ridge_alpha": float(ridge.alpha), "refit_on_train_and_val": False}
        save_json(output / "selection.json", selected)
        stage, predictions = "prediction_and_evaluation", {}
        with threadpool_limits(limits=4):
            for split in ["val", "test"]:
                predictions[("Persistence", split)] = data[f"last_co2_{split}"].copy()
                predictions[("Ridge", split)] = ridge.predict(flatten(data[f"X_{split}"]))
            recovered = joblib.load(output / "model.joblib")
            np.testing.assert_allclose(recovered["ridge"].predict(flatten(data["X_test"])), predictions[("Ridge", "test")], rtol=1e-12)
        table, detail, periods = evaluate(data, predictions, boundaries["test"]["start_timestamp"])
        table.to_csv(output / "metrics.csv", index=False)
        detail.to_csv(output / "predictions.csv", index=False)
        periods.to_csv(output / "metrics_by_period.csv", index=False)
        stage = "plotting"
        plot_results(data, predictions, table, boundaries["test"]["start_timestamp"], output / "figures")
        stage = "final_verification"
        for name, digest in input_hashes.items():
            if sha256(prepared_dir / name) != digest:
                raise DataError("预处理产物被改变", file=name)
        if sha256(Path(source["input"])) != source["data_sha256"]:
            raise DataError("原始数据被改变")
        saved = pd.read_csv(output / "predictions.csv", converters={"y_true": float, "y_pred": float})
        for record in table.to_dict("records"):
            part = saved[(saved["model"] == record["model"]) & (saved["split"] == record["split"])]
            split = record["split"]
            np.testing.assert_array_equal(part["target_original_row"], data[f"target_original_rows_{split}"])
            np.testing.assert_array_equal(pd.to_datetime(part["target_timestamp"]).to_numpy(dtype="datetime64[ns]"), data[f"target_timestamps_{split}"])
            for metric, value in metrics(part["y_true"], part["y_pred"]).items():
                np.testing.assert_allclose(value, record[metric], rtol=1e-12)
        config["status"] = "complete"
        save_json(output / "config.json", config)
        summary = {"status": "complete", "selection": selected,
                   "metrics": table.astype(object).where(pd.notna(table), None).to_dict("records"),
                   "timing": {"ridge_search_seconds": search_seconds}, "elapsed_seconds": time.perf_counter() - started,
                   "checks": {"model_reload_predictions_match": True, "prediction_csv_metrics_match": True,
                              "source_and_preprocessing_unchanged": True, "shared_targets_verified": True}}
        save_json(output / "summary.json", summary)
        print(table.to_string(index=False), flush=True)
        print(f"Ridge 实验完成：{output}", flush=True)
        return summary
    except Exception as exc:
        if config is not None:
            config["status"] = "failed"
            save_json(output / "config.json", config)
        save_json(output / "failure.json", {"stage": stage, "exception": type(exc).__name__,
                  "message": str(exc), "details": getattr(exc, "details", {}), "traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=ROOT / "results" / "prepare_final")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / datetime.now(timezone.utc).strftime("ridge_%Y%m%dT%H%M%S_%fZ"))
    args = parser.parse_args()
    try:
        run(args.prepared_dir, args.output_dir)
    except Exception as exc:
        print(f"实验失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
