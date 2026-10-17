import contextlib
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import train


class DataLoadingTests(unittest.TestCase):
    def setUp(self):
        self.signals = np.zeros((6, 32), dtype=np.float32)
        self.subjects = np.repeat(["patient-a", "patient-b", "patient-c"], 2)
        self.module = types.ModuleType("custom_data")
        self.module.load = Mock(return_value=(self.signals, self.signals, self.subjects))

    def test_custom_loader_cli_runs_without_importing_alyx(self):
        # Blocking the Alyx module proves another loader does not depend on it.
        with patch.dict(sys.modules, {"custom_data": self.module, "alyx_data": None}), patch.object(
            sys, "argv", ["train.py", "--data-loader", "custom_data:load", "--data-dir", "other-data",
                          "--sampling-rate", "64", "--window-seconds", "0.5",
                          "--held-out-subject", "patient-c", "--check-data"]
        ), contextlib.redirect_stdout(io.StringIO()) as output:
            train.main()
        self.module.load.assert_called_once_with(
            data_dir=Path("other-data"), sampling_rate=64, window_seconds=0.5, overlap_ratio=0.2)
        self.assertIn("test: 2 windows, subjects ['patient-c']", output.getvalue())
        self.assertIn("train: 2 windows", output.getvalue())
        self.assertIn("validation: 2 windows", output.getvalue())

    def test_rejects_mismatched_pairs_or_subjects_and_nonfinite_samples(self):
        bad_signals = self.signals.copy()
        bad_signals[0, 0] = np.nan
        for result in ((self.signals, self.signals[:-1], self.subjects),
                       (self.signals, self.signals, self.subjects[:-1]),
                       (bad_signals, self.signals, self.subjects),
                       (self.signals + 2, self.signals, self.subjects)):
            with self.subTest(result=result), patch.dict(sys.modules, {"custom_data": self.module}):
                self.module.load.return_value = result
                with self.assertRaises(ValueError):
                    train.load_training_data("custom_data:load", None, sampling_rate=64,
                                             window_seconds=0.5, overlap_ratio=0.2)

    def test_numeric_subject_id_from_cli_selects_correct_test_subject(self):
        subjects = np.repeat([42, 43, 44], 2)
        train_indices, val_indices, test_indices = train.split_subjects(subjects, "44")
        np.testing.assert_array_equal(subjects[test_indices], [44, 44])
        self.assertFalse(set(subjects[train_indices]) & set(subjects[val_indices]))


if __name__ == "__main__":
    unittest.main()
