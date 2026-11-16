import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import backfill_test_metrics as backfill
import eval as evaluation
from train import ModelConfig


class BackfillTests(unittest.TestCase):
    def test_run_id_has_priority_and_conflicting_id_is_not_matched_by_name(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root) / "run_name"
            (run_dir / "wandb" / "run-20261104_200933-abc123").mkdir(parents=True)
            (run_dir / "best_model.pt").touch()
            by_id, by_name = backfill.index_checkpoints(Path(root))
            self.assertEqual(backfill.checkpoint_for_run(
                SimpleNamespace(id="abc123", name="renamed"), by_id, by_name),
                run_dir / "best_model.pt")
            self.assertIsNone(backfill.checkpoint_for_run(
                SimpleNamespace(id="other", name="run_name"), by_id, by_name))

    def test_config_mismatch_blocks_wrong_checkpoint(self):
        cfg = ModelConfig(data_fraction=0.1, seed=7)
        run = SimpleNamespace(config={"data_fraction": 1.0, "seed": 7})
        matched, reason = backfill.matches_wandb_config(run, cfg)
        self.assertFalse(matched)
        self.assertIn("data_fraction", reason)

    def test_frechet_handles_full_window_without_recursion(self):
        signal = np.sin(np.arange(512) / 20)
        self.assertEqual(evaluation.discrete_frechet_distance(signal, signal), 0.0)


if __name__ == "__main__":
    unittest.main()
