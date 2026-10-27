"""验证会影响模型选择、公平评价与产物恢复的关键行为。"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

import run_experiment as exp
from prepare_data import DataError, run as prepare, sha256
from test_prepare_data import sequence, write_csv


class EvaluationTests(unittest.TestCase):
    def test_metrics_formula_and_shape_guard(self):
        score = exp.metrics([1, 2, 3], [2, 2, 1])
        self.assertEqual(score["mae"], 1)
        self.assertAlmostEqual(score["rmse"], np.sqrt(5 / 3))
        for y, prediction in [([], []), ([1], [[1]]), ([[1]], [1]), ([1, 2], [1]),
                              ([np.nan], [1]), ([1], [np.inf])]:
            with self.subTest(y=y, prediction=prediction), self.assertRaises(DataError):
                exp.metrics(y, prediction)

    def test_improvement_zero_and_regression(self):
        self.assertIsNone(exp.improvement(0, 0))
        self.assertIsNone(exp.improvement(0, 1))
        self.assertEqual(exp.improvement(2, 3), -50)
        self.assertEqual(exp.improvement(2, 1), 50)

    def test_ridge_exact_validation_selection_and_ties(self):
        records = [{"alpha": 100, "rmse": 1.00000001}, {"alpha": 0.1, "rmse": 1.00000002}]
        self.assertEqual(exp.best_ridge_index(records), 0)
        records[1]["rmse"] = records[0]["rmse"]
        self.assertEqual(exp.best_ridge_index(records), 1)

    def test_flatten_time_then_features(self):
        x = np.arange(2 * 12 * 21).reshape(2, 12, 21)
        flat = exp.flatten(x)
        np.testing.assert_array_equal(flat[0], np.arange(252))
        self.assertEqual(flat[1, 21], x[1, 1, 0])
        with self.assertRaises(DataError):
            exp.flatten(np.zeros((2, 21, 12)))

    def test_gap_keeps_both_endpoints(self):
        stamps = np.array(["2020-01-01T00:00", "2020-01-01T00:10", "2020-01-01T00:40"], dtype="datetime64[ns]")
        x, y = exp.gap_series(stamps, [1, 2, 3])
        self.assertEqual(len(y), 4)
        self.assertTrue(np.isnan(y[2]))
        np.testing.assert_array_equal(y[[0, 1, 3]], [1, 2, 3])
        np.testing.assert_array_equal(x[[0, 1, 3]], stamps)

    def test_shared_targets_zero_baseline_and_seven_day_bins(self):
        stamps = np.array(["2020-01-01T00:00", "2020-01-01T00:10", "2020-01-08T00:00"], dtype="datetime64[ns]")
        data, predictions = {}, {}
        for split in ["val", "test"]:
            data.update({f"y_{split}": np.array([10., 20., 30.]), f"target_timestamps_{split}": stamps,
                         f"target_original_rows_{split}": np.array([12, 13, 100])})
            for method in exp.METHODS:
                predictions[(method, split)] = data[f"y_{split}"].copy() + (method != "Persistence")
        table, detail, periods = exp.evaluate(data, predictions, str(stamps[0]))
        self.assertEqual(len(table), 4)
        self.assertTrue(table["mae_improvement_pct"].isna().all())
        self.assertTrue(table["improvement_note"].str.contains("zero").all())
        self.assertEqual(periods[periods.model == "Ridge"]["n"].tolist(), [2, 1])
        for split in ["val", "test"]:
            for method in exp.METHODS:
                part = detail[(detail.split == split) & (detail.model == method)]
                np.testing.assert_array_equal(part.target_original_row, data[f"target_original_rows_{split}"])


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw" / "weather.csv"
        _, stamps, values, _ = sequence()
        write_csv(self.raw, stamps, values)
        self.prepared = self.root / "prepared"
        prepare(self.raw, self.prepared)

    def test_prepared_shape_order_units_and_source_hash_validation(self):
        data, _, _, _ = exp.load_prepared(self.prepared)
        self.assertEqual(data["X_test"].shape, (15, 12, 21))
        data["last_co2_test"] /= 100
        np.savez_compressed(self.prepared / "dataset.npz", **data)
        with self.assertRaises(AssertionError):
            exp.load_prepared(self.prepared)
        self.raw.write_text(self.raw.read_text() + "\n")
        with self.assertRaisesRegex(DataError, "哈希"):
            exp.load_prepared(self.prepared)

    def test_complete_pipeline_reload_and_refuse_overwrite(self):
        output = self.root / "experiment"
        before = sha256(self.raw)
        with contextlib.redirect_stdout(io.StringIO()):
            summary = exp.run(self.prepared, output)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(sha256(self.raw), before)
        self.assertTrue(all(summary["checks"].values()))
        search = pd.read_csv(output / "validation_search.csv")
        self.assertEqual(len(search), 5)
        self.assertEqual(search.selected.sum(), 1)
        bundle = joblib.load(output / "model.joblib")
        self.assertEqual(bundle["ridge"].n_features_in_, 252)
        self.assertEqual(len(list((output / "figures").glob("*.png"))), 4)
        self.assertEqual(len(list((output / "figures").glob("*.svg"))), 4)
        before_config = (output / "config.json").read_bytes()
        with self.assertRaises(FileExistsError):
            exp.run(self.prepared, output)
        self.assertEqual((output / "config.json").read_bytes(), before_config)

    def test_failed_inputs_saved_with_stage(self):
        output = self.root / "failed"
        (self.prepared / "dataset.npz").unlink()
        with self.assertRaises(FileNotFoundError):
            exp.run(self.prepared, output)
        error = json.loads((output / "failure.json").read_text())
        self.assertEqual(error["stage"], "input_checks")


if __name__ == "__main__":
    unittest.main()
