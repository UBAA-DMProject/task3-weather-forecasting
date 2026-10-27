"""用可核对的小序列验证时间窗口、标准化范围与产物安全性。"""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from prepare_data import (
    DataError, FEATURES, LENGTH, build_windows, fixed_splits, load_checked,
    make_arrays, run, sha256, standardize,
)


def sequence(n=100):
    rows = np.arange(n)
    stamps = np.datetime64("2020-01-01T00:10", "ns") + rows * np.timedelta64(10, "m")
    values = rows[:, None] * 100.0 + np.arange(21)[None, :]
    return rows, stamps, values, np.zeros_like(values, dtype=bool)


def write_csv(path, stamps, values, header=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header or ["date", *FEATURES])
        for stamp, row in zip(stamps, values):
            writer.writerow([pd.Timestamp(stamp).strftime("%Y/%m/%d %H:%M"), *row])


class WindowTests(unittest.TestCase):
    def test_alignment_boundary_context_and_original_units(self):
        rows, stamps, values, invalid = sequence()
        splits = fixed_splits(len(rows))
        targets, windows = build_windows(rows, stamps, values, invalid, splits)
        self.assertEqual({k: len(v) for k, v in targets.items()}, {"train": 58, "val": 15, "test": 15})
        self.assertEqual(targets["train"][0], 12)
        self.assertEqual(targets["val"][0], 70)
        self.assertEqual(targets["test"][0], 85)
        self.assertEqual(windows.loc[70, "input_0_original_row"], 58)
        self.assertEqual(windows.loc[85, "input_11_original_row"], 84)
        scaler, fit, _ = standardize(values, invalid, splits["train"][1])
        arrays = make_arrays(rows, stamps, values, targets, scaler)
        restored = scaler.inverse_transform(arrays["X_train"][0])
        np.testing.assert_allclose(restored, values[:12], atol=1e-10)
        np.testing.assert_allclose(restored.reshape(252), values[:12].reshape(252), atol=1e-10)
        self.assertEqual(arrays["y_train"].shape, (58,))
        self.assertEqual(arrays["y_train"][0], values[12, -1])
        self.assertEqual(arrays["last_co2_train"][0], values[11, -1])
        np.testing.assert_array_equal(arrays["target_timestamps_test"], stamps[85:])
        np.testing.assert_array_equal(fit, np.arange(70))

    def test_invalid_target_other_feature_and_overlapping_reasons(self):
        rows, stamps, values, invalid = sequence()
        invalid[20, 0] = True
        invalid[40, -1] = True
        stamps[41:] += np.timedelta64(90, "m")
        targets, windows = build_windows(rows, stamps, values, invalid, fixed_splits(100))
        self.assertIn(20, targets["train"])
        self.assertTrue(windows.loc[40, "invalid_target"])
        for j in range(21, 33):
            self.assertTrue(windows.loc[j, "invalid_input"])
        for j in range(41, 53):
            self.assertTrue(windows.loc[j, "invalid_input"])
            self.assertTrue(windows.loc[j, "time_gap"])
        self.assertTrue(windows.loc[53, "valid"])
        self.assertGreater(sum(windows[r].sum() for r in
                               ["insufficient_history", "time_gap", "invalid_input", "invalid_target"]),
                           (~windows["valid"]).sum())

    def test_every_interval_including_target_is_checked(self):
        for gap_position in range(1, LENGTH + 1):
            with self.subTest(gap_position=gap_position):
                rows, stamps, values, invalid = sequence()
                stamps[gap_position:] += np.timedelta64(10, "m")
                _, windows = build_windows(rows, stamps, values, invalid, fixed_splits(100))
                self.assertTrue(windows.loc[12, "time_gap"])

    def test_scaler_unique_valid_training_rows_and_constant_column(self):
        _, _, values, invalid = sequence()
        values[:, 0] = 5
        invalid[10, 1] = True
        values[10, 1] = -9999
        values[70:] = 1e12
        scaler, fit, constants = standardize(values, invalid, 70)
        expected = np.delete(values[:70], 10, axis=0)
        self.assertEqual(scaler.n_samples_seen_, 69)
        np.testing.assert_allclose(scaler.mean_, expected.mean(axis=0))
        self.assertNotIn(10, fit)
        self.assertEqual(constants, [FEATURES[0]])
        self.assertEqual(scaler.scale_[0], 1)
        invalid[:70] = True
        with self.assertRaises(DataError):
            standardize(values, invalid, 70)

    def test_empty_split_refused(self):
        rows, stamps, values, invalid = sequence()
        invalid[85:, -1] = True
        targets, _ = build_windows(rows, stamps, values, invalid, fixed_splits(100))
        scaler, _, _ = standardize(values, invalid, 70)
        with self.assertRaisesRegex(DataError, "没有有效窗口"):
            make_arrays(rows, stamps, values, targets, scaler)


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "raw" / "weather.csv"
        self.output = self.root / "results" / "run"
        self.rows, self.stamps, self.values, _ = sequence()

    def test_duplicate_invalid_cells_and_no_time_axis_deletion(self):
        stamps = np.insert(self.stamps, 20, self.stamps[19])
        values = np.insert(self.values.astype(object), 20, self.values[19], axis=0)
        values[30, 0:5] = ["", "oops", "inf", "NaN", "-9999"]
        write_csv(self.input, stamps, values)
        self.output.mkdir(parents=True)
        rows, _, _, invalid, checks = load_checked(self.input, self.output)
        self.assertEqual(len(rows), 100)
        self.assertIn(30, rows)
        self.assertEqual(invalid.sum(), 5)
        self.assertEqual(checks["invalid_by_reason_original"],
                         {"empty": 1, "non_numeric": 1, "non_finite": 2, "sentinel_minus_9999": 1})
        mapping = pd.read_csv(self.output / "duplicate_mapping.csv")
        self.assertEqual(mapping.loc[0, "deleted_original_row"], 20)
        self.assertEqual(mapping.loc[0, "retained_original_row"], 19)

    def test_duplicate_conflict_time_reversal_and_bad_timestamp(self):
        for case in ["conflict", "backwards", "parse"]:
            with self.subTest(case=case):
                stamps, values = self.stamps.copy(), self.values.copy()
                stamps[20] = stamps[19] if case == "conflict" else stamps[18]
                write_csv(self.input, stamps, values)
                if case == "parse":
                    self.input.write_text(self.input.read_text().replace("2020/01/01 00:10", "bad-time"))
                output = self.root / "results" / case
                with self.assertRaises(DataError):
                    run(self.input, output)
                failure = json.loads((output / "failure.json").read_text())
                self.assertEqual(failure["stage"], "load_and_check")
                self.assertTrue(failure["details"])

    def test_header_order_and_malformed_rows(self):
        header = ["date", *FEATURES]
        header[1], header[2] = header[2], header[1]
        write_csv(self.input, self.stamps, self.values, header)
        with self.assertRaisesRegex(DataError, "字段或顺序"):
            run(self.input, self.output)
        write_csv(self.input, self.stamps, self.values)
        with self.input.open("a") as stream:
            stream.write("too,few,columns\n")
        with self.assertRaisesRegex(DataError, "列数错误"):
            run(self.input, self.root / "results" / "malformed")

    def test_roundtrip_unchanged_source_and_refuse_overwrite(self):
        write_csv(self.input, self.stamps, self.values)
        before = sha256(self.input)
        summary = run(self.input, self.output)
        self.assertEqual(sha256(self.input), before)
        self.assertTrue(summary["saved_arrays_verified"])
        with np.load(self.output / "dataset.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["y_test"], self.values[85:, -1])
            scaler = joblib.load(self.output / "scaler.joblib")
            np.testing.assert_allclose(scaler.inverse_transform(saved["X_test"][0]),
                                       self.values[73:85], atol=1e-10)
            all_targets = np.concatenate([saved[f"target_original_rows_{s}"] for s in ["train", "val", "test"]])
            self.assertTrue(np.all(np.diff(all_targets) > 0))
        config_before = (self.output / "config.json").read_bytes()
        with self.assertRaises(FileExistsError):
            run(self.input, self.output)
        self.assertEqual((self.output / "config.json").read_bytes(), config_before)
        with self.assertRaisesRegex(DataError, "原始数据目录"):
            run(self.input, self.input.parent / "output")
        self.assertFalse((self.input.parent / "output").exists())


if __name__ == "__main__":
    unittest.main()
