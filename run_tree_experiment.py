"""探索性改进：仅按验证集选择非线性 CO₂ 增量模型。"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from prepare_data import FEATURES, LENGTH, ROOT, DataError, save_json, sha256
from run_experiment import (
    SNAPSHOTS, evaluate, flatten, gap_series, load_prepared, metrics,
    plt, save_figure, train_ridge,
)


def candidates():
    return [
        {"family": "HistGradientBoosting", "learning_rate": lr, "max_leaf_nodes": leaves,
         "max_iter": 400, "min_samples_leaf": 30, "l2_regularization": 10.0}
        for lr in (0.05, 0.1) for leaves in (7, 15, 31)
    ]


def tree_features(x):
    """完整历史 252 维 + 相邻差分 231 维；不接收目标或未来观测。"""
    levels = flatten(x)
    changes = np.diff(x, axis=1).reshape(len(x), (LENGTH - 1) * len(FEATURES))
    return np.concatenate([levels, changes], axis=1)


def predict_delta(model, features, iteration):
    if iteration is None:
        return model.predict(features)
    for i, prediction in enumerate(model.staged_predict(features), 1):
        if i == iteration:
            return prediction
    raise DataError("梯度提升检查点轮次不存在", iteration=iteration)


def predict_tree(bundle, standardized_x):
    features = tree_features(standardized_x)
    scaler = bundle["scaler"]
    last_co2 = standardized_x[:, -1, -1] * scaler.scale_[-1] + scaler.mean_[-1]
    return last_co2 + predict_delta(bundle["estimator"], features, bundle["best_iteration"])


def search_trees(x_train, y_train, last_train, x_val, y_val, last_val, output, grid=None):
    """接口只接收训练、验证数据；所有 ppm 指标在加回持续值后计算。"""
    grid = candidates() if grid is None else grid
    train, val = tree_features(x_train), tree_features(x_val)
    delta_train, delta_val = y_train - last_train, y_val - last_val
    best_model, best_record, records, history = None, None, [], []
    for number, spec in enumerate(grid, 1):
        params = {key: value for key, value in spec.items() if key != "family"}
        started = time.perf_counter()
        if spec["family"] == "HistGradientBoosting":
            model = HistGradientBoostingRegressor(
                **params, loss="squared_error", early_stopping=True,
                scoring="neg_root_mean_squared_error", validation_fraction=None,
                n_iter_no_change=20, tol=1e-7, random_state=42,
            ).fit(train, delta_train, X_val=val, y_val=delta_val)
            best_stage, score = None, None
            for epoch, delta in enumerate(model.staged_predict(val), 1):
                current = metrics(y_val, last_val + delta)
                history.append({"candidate": number, "iteration": epoch, **current})
                if score is None or current["rmse"] < score["rmse"]:
                    best_stage, score = epoch, current
            trained_iterations = int(model.n_iter_)
        else:
            raise DataError("未知树模型类型", family=spec["family"])
        record = {"candidate": number, "family": spec["family"], "parameters": json.dumps(params, sort_keys=True),
                  "best_iteration": best_stage, "trained_iterations": trained_iterations,
                  **score, "fit_and_validation_seconds": time.perf_counter() - started}
        records.append(record)
        # 精确平局时保留预先登记的较早候选；不使用测试分数。
        if best_record is None or record["rmse"] < best_record["rmse"]:
            best_model, best_record = model, record
        pd.DataFrame(records).to_csv(output / "tree_validation_search.csv", index=False)
        pd.DataFrame(history).to_csv(output / "boosting_history.csv", index=False)
        print(f"Tree {number}/{len(grid)} {spec['family']}: validation RMSE={record['rmse']:.6f} ppm, "
              f"time={record['fit_and_validation_seconds']:.1f}s", flush=True)
    if best_record is None:
        raise DataError("没有树模型候选")
    for record in records:
        record["selected"] = record["candidate"] == best_record["candidate"]
    pd.DataFrame(records).to_csv(output / "tree_validation_search.csv", index=False)
    return best_model, best_record, records


def plots(data, predictions, table, methods, test_start, output):
    output.mkdir()
    stamps, truth = data["target_timestamps_test"], data["y_test"]
    start = np.datetime64(test_start, "ns")
    for name, mask in [("test_predictions", np.ones(len(truth), dtype=bool)),
                       ("test_first_24h", (stamps >= start) & (stamps < start + np.timedelta64(24, "h")))]:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(*gap_series(stamps[mask], truth[mask]), label="Observed", color="black", linewidth=1)
        for method in methods:
            ax.plot(*gap_series(stamps[mask], predictions[(method, "test")][mask]), label=method, linewidth=0.8, alpha=0.8)
        ax.set(title="Exploratory one-step CO2 predictions", xlabel="Target time", ylabel="CO2 (ppm)")
        ax.legend(ncol=4)
        ax.grid(alpha=0.2)
        fig.autofmt_xdate()
        save_figure(fig, output, name)
    fig, axes = plt.subplots(len(methods), 1, figsize=(12, 7), sharex=True)
    for method, ax in zip(methods, axes):
        ax.plot(*gap_series(stamps, predictions[(method, "test")] - truth), linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set(title=method, ylabel="Residual (ppm)")
    axes[-1].set_xlabel("Target time")
    save_figure(fig, output, "test_residuals")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    test = table[table.split == "test"].set_index("model").loc[list(methods)]
    for ax, metric in zip(axes, ["mae", "rmse"]):
        bars = ax.bar(methods, test[metric])
        ax.bar_label(bars, fmt="%.4f")
        ax.set(title=f"Exploratory test {metric.upper()}", ylabel="Error (ppm)", ylim=(0, test[metric].max() * 1.2))
    save_figure(fig, output, "test_error_comparison")


def run(prepared_dir, output):
    prepared_dir, output = Path(prepared_dir).resolve(), Path(output).resolve()
    if output == prepared_dir or prepared_dir in output.parents or output == ROOT / "dataset" or ROOT / "dataset" in output.parents:
        raise DataError("输出不能位于原始数据或预处理目录")
    output.mkdir(parents=True, exist_ok=False)
    config, stage, started = None, "input_checks", time.perf_counter()
    try:
        source_hashes = {name: sha256(prepared_dir / name) for name in [*SNAPSHOTS, "dataset.npz", "scaler.joblib"]}
        data, scaler, source, boundaries = load_prepared(prepared_dir)
        config = {"status": "running", "exploratory": True,
                  "reason": "基于已见测试结果提出新模型；原测试集已不构成新的独立检验",
                  "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "command": shlex.join([sys.executable, *sys.argv]), "working_directory": str(Path.cwd()),
                  "prepared_dir": str(prepared_dir), "source_data_sha256": source["data_sha256"],
                  "input_artifact_sha256": source_hashes,
                  "code_sha256": {name: sha256(ROOT / name) for name in
                                  ["run_tree_experiment.py", "test_tree_experiment.py", "run_experiment.py", "prepare_data.py", "requirements.txt"]},
                  "candidate_grid": candidates(), "seed": 42, "threads": 4,
                  "input_features": "252 standardized levels + 231 adjacent historical differences",
                  "tree_target": "next CO2 minus last observed CO2, in ppm",
                  "selection_rule": "unrounded validation RMSE; exact tie uses earlier candidate",
                  "refit_on_train_and_val": False}
        save_json(output / "config.json", config)
        save_json(output / "environment.json", {"python": sys.version, "platform": platform.platform(),
                  "packages": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "matplotlib", "threadpoolctl"]}})
        (output / "preprocessing").mkdir()
        for name in SNAPSHOTS:
            shutil.copyfile(prepared_dir / name, output / "preprocessing" / name)
        stage = "validation_search"
        with threadpool_limits(limits=4):
            ridge, ridge_records = train_ridge(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
            pd.DataFrame(ridge_records).to_csv(output / "ridge_validation_search.csv", index=False)
            tree, selected, search = search_trees(
                data["X_train"], data["y_train"], data["last_co2_train"],
                data["X_val"], data["y_val"], data["last_co2_val"], output)
        method = selected["family"] + "Delta"
        bundle = {"estimator": tree, "best_iteration": selected["best_iteration"], "scaler": scaler,
                  "feature_names": FEATURES, "window_length": LENGTH, "feature_mode": "levels_and_differences",
                  "target_transform": "delta_from_last_co2", "target_unit": "ppm", "exploratory": True}
        joblib.dump(bundle, output / "tree_model.joblib")
        joblib.dump({"ridge": ridge, "scaler": scaler, "feature_names": FEATURES, "window_length": LENGTH}, output / "ridge_model.joblib")
        ridge_val = metrics(data["y_val"], ridge.predict(flatten(data["X_val"])))
        persist_val = metrics(data["y_val"], data["last_co2_val"])
        recommended = min([(persist_val["rmse"], "Persistence"), (ridge_val["rmse"], "Ridge"), (selected["rmse"], method)], key=lambda r: r[0])[1]
        selection = {"tree": selected, "recommended_by_validation": recommended, "ridge_alpha": float(ridge.alpha),
                     "exploratory": True, "test_used_for_selection": False}
        save_json(output / "selection.json", selection)
        # 以上选参和模型保存完成后才预测测试集；未选中的树候选不计算测试指标。
        stage, predictions = "evaluation", {}
        with threadpool_limits(limits=4):
            for split in ["val", "test"]:
                predictions[("Persistence", split)] = data[f"last_co2_{split}"].copy()
                predictions[("Ridge", split)] = ridge.predict(flatten(data[f"X_{split}"]))
                predictions[(method, split)] = predict_tree(bundle, data[f"X_{split}"])
            restored = joblib.load(output / "tree_model.joblib")
            np.testing.assert_allclose(predict_tree(restored, data["X_test"]), predictions[(method, "test")], rtol=1e-12)
        methods = ("Persistence", "Ridge", method)
        table, detail, periods = evaluate(data, predictions, boundaries["test"]["start_timestamp"], methods)
        for frame in [table, detail, periods]:
            frame["exploratory"] = True
        table.to_csv(output / "metrics.csv", index=False)
        detail.to_csv(output / "predictions.csv", index=False)
        periods.to_csv(output / "metrics_by_period.csv", index=False)
        stage = "plots_and_checks"
        plots(data, predictions, table, methods, boundaries["test"]["start_timestamp"], output / "figures")
        for name, digest in source_hashes.items():
            if sha256(prepared_dir / name) != digest:
                raise DataError("预处理输入被修改", file=name)
        if sha256(Path(source["input"])) != source["data_sha256"]:
            raise DataError("原始数据被修改")
        saved = pd.read_csv(output / "predictions.csv", converters={"y_true": float, "y_pred": float})
        for split in ["val", "test"]:
            for name in methods:
                part = saved[(saved.model == name) & (saved.split == split)]
                np.testing.assert_array_equal(part.target_original_row, data[f"target_original_rows_{split}"])
                np.testing.assert_array_equal(pd.to_datetime(part.target_timestamp).to_numpy(dtype="datetime64[ns]"), data[f"target_timestamps_{split}"])
                score = metrics(part.y_true, part.y_pred)
                recorded = table[(table.model == name) & (table.split == split)].iloc[0]
                for metric, value in score.items():
                    np.testing.assert_allclose(value, recorded[metric], rtol=1e-12)
        config["status"] = "complete"
        save_json(output / "config.json", config)
        summary = {"status": "complete", "exploratory": True, "selection": selection,
                   "metrics": table.astype(object).where(pd.notna(table), None).to_dict("records"),
                   "elapsed_seconds": time.perf_counter() - started,
                   "checks": {"model_reload_matches": True, "prediction_metrics_and_targets_match": True,
                              "source_and_preprocessing_unchanged": True}}
        save_json(output / "summary.json", summary)
        print(table.to_string(index=False), flush=True)
        print(f"探索性实验完成：{output}", flush=True)
        return summary
    except Exception as exc:
        if config is not None:
            config["status"] = "failed"
            save_json(output / "config.json", config)
        save_json(output / "failure.json", {"stage": stage, "message": str(exc),
                  "details": getattr(exc, "details", {}), "traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=ROOT / "results" / "prepare_final")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / datetime.now(timezone.utc).strftime("tree_exploration_%Y%m%dT%H%M%S_%fZ"))
    args = parser.parse_args()
    try:
        run(args.prepared_dir, args.output_dir)
    except Exception as exc:
        print(f"树模型实验失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
