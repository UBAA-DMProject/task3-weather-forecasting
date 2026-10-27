"""非线性增量模型的目标恢复、验证检查点及运行产物检查。"""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler

from prepare_data import run as prepare, sha256
import run_tree_experiment as tree
from test_prepare_data import sequence, write_csv


class TreeTests(unittest.TestCase):
    def test_features_preserve_all_levels_and_historical_differences(self):
        x = np.arange(2 * 12 * 21).reshape(2, 12, 21)
        features = tree.tree_features(x)
        self.assertEqual(features.shape, (2, 483))
        np.testing.assert_array_equal(features[0, :252], np.arange(252))
        np.testing.assert_array_equal(features[:, 252:], np.full((2, 231), 21))

    def test_delta_prediction_restored_to_ppm(self):
        raw = np.zeros((2, 21))
        raw[:, -1] = [390, 410]
        scaler = StandardScaler().fit(raw)
        standardized = np.zeros((3, 12, 21))
        standardized[:, -1, -1] = [0, 1, -1]
        model = DummyRegressor(strategy="constant", constant=2).fit(np.zeros((2, 483)), [2, 2])
        bundle = {"estimator": model, "best_iteration": None, "scaler": scaler}
        np.testing.assert_array_equal(tree.predict_tree(bundle, standardized), [402, 412, 392])

    def test_selected_boosting_prefix_not_last_iteration(self):
        class Stages:
            def staged_predict(self, x):
                yield np.full(len(x), 1.)
                yield np.full(len(x), 2.)
                yield np.full(len(x), 3.)
        prediction = tree.predict_delta(Stages(), np.zeros((4, 483)), 2)
        np.testing.assert_array_equal(prediction, [2, 2, 2, 2])

    def test_search_supplies_external_validation_and_restores_best_stage(self):
        class FakeBoosting:
            def __init__(self, **kwargs):
                self.n_iter_ = 3
            def fit(self, x, y, *, X_val, y_val):
                np.testing.assert_array_equal(y, np.full(30, 5.))
                np.testing.assert_array_equal(y_val, np.full(8, 2.))
                return self
            def staged_predict(self, x):
                yield np.full(len(x), 1.)
                yield np.full(len(x), 2.)
                yield np.full(len(x), 3.)
        grid = [{"family": "HistGradientBoosting", "max_iter": 3}]
        with tempfile.TemporaryDirectory() as directory, patch.object(tree, "HistGradientBoostingRegressor", FakeBoosting), contextlib.redirect_stdout(io.StringIO()):
            _, selected, _ = tree.search_trees(np.zeros((30, 12, 21)), np.full(30, 405.), np.full(30, 400.),
                                              np.zeros((8, 12, 21)), np.full(8, 402.), np.full(8, 400.), Path(directory), grid)
        self.assertEqual(selected["best_iteration"], 2)
        self.assertEqual(selected["rmse"], 0)

    def test_pipeline_source_unchanged_and_reload(self):
        grid = [
            {"family": "HistGradientBoosting", "max_iter": 3, "max_leaf_nodes": 7},
            {"family": "HistGradientBoosting", "max_iter": 3, "max_leaf_nodes": 3},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw" / "weather.csv"
            _, stamps, values, _ = sequence()
            write_csv(raw, stamps, values)
            prepared, output = root / "prepared", root / "tree_run"
            prepare(raw, prepared)
            before = sha256(raw)
            with patch.object(tree, "candidates", return_value=grid), contextlib.redirect_stdout(io.StringIO()):
                summary = tree.run(prepared, output)
            self.assertTrue(summary["exploratory"])
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(sha256(raw), before)
            self.assertTrue(all(summary["checks"].values()))
            self.assertEqual(len(list((output / "figures").glob("*.png"))), 4)
            bundle = joblib.load(output / "tree_model.joblib")
            self.assertEqual(bundle["target_transform"], "delta_from_last_co2")
            with self.assertRaises(FileExistsError):
                tree.run(prepared, output)


if __name__ == "__main__":
    unittest.main()
