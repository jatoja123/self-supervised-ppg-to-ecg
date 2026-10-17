import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import train
import alyx_data


class AlyxTrainingTests(unittest.TestCase):
    def test_timestamps_are_sorted_and_units_converted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ECG.json"
            for multiplier in (1, 1000, 1_000_000, 1_000_000_000):
                path.write_text(json.dumps({"data": {
                    str(1_650_000_001 * multiplier): [2],
                    str(1_650_000_000 * multiplier): [1],
                }}))
                times, values = alyx_data.read_alyx_signal(path)
                np.testing.assert_allclose(times - times[0], [0, 1])
                np.testing.assert_array_equal(values, [1, 2])
            path.write_text(json.dumps({"data": {
                "2022-05-19T10:00:01Z": [2], "2022-05-19T10:00:00Z": [1],
            }}))
            times, _ = alyx_data.read_alyx_signal(path)
            self.assertEqual(times[1] - times[0], 1)

    @staticmethod
    def write_session(root, player, gap=False, indexed=False):
        for device, name, fs, offset in (("polar-h10", "ECG", 130, 0), ("empatica", "PPG", 64, 0.5)):
            path = root / "players" / str(player) / "2022-05-19" / device / f"{name}.json"
            path.parent.mkdir(parents=True)
            t = np.arange(fs * 16) / fs + offset
            if gap and name == "PPG":
                t = t[(t < 7) | (t > 10)]
            x = np.sin(2 * np.pi * 1.2 * t)
            if indexed:
                # Receipt timestamps are shared by batches of samples.
                data = {
                    str(round((time - offset) * fs) + 1): [
                        float(value), (datetime(2022, 5, 19, 10, tzinfo=timezone.utc)
                                       + timedelta(seconds=offset + i // fs)).isoformat()]
                    for i, (time, value) in enumerate(zip(t, x))
                }
            else:
                data = {str((1_650_000_000 + time) * 1000): [float(value)]
                        for time, value in zip(t, x)}
            path.write_text(json.dumps({"data": data}))

    def test_indexed_batch_recordings_keep_samples_and_native_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_session(root, 42, indexed=True)
            path = root / "players/42/2022-05-19/empatica/PPG.json"
            times, values = alyx_data.read_alyx_signal(path)
            self.assertEqual(len(times), 64 * 16)
            np.testing.assert_allclose(np.diff(times), 1 / 64)
            self.assertEqual(len(values), len(times))
            ppg, ecg, subjects = alyx_data.load_alyx_segments(root)
            self.assertEqual(ppg.shape, (4, 520))
            self.assertEqual(ecg.shape, ppg.shape)
            self.assertEqual(set(subjects), {42})
            self.assertGreater(np.corrcoef(ppg[1], ecg[1])[0, 1], 0.98)

    def test_indexed_recordings_preserve_missing_sample_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_session(root, 42, gap=True, indexed=True)
            with self.assertRaisesRegex(ValueError, "No usable paired windows"):
                alyx_data.load_alyx_segments(root)

    def test_loader_aligns_resamples_normalizes_and_splits_players(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for player in (42, 43, 44):
                self.write_session(root, player)
            ppg, ecg, subjects = alyx_data.load_alyx_segments(root, sampling_rate=130)
            self.assertEqual(ppg.shape, (12, 520))
            self.assertEqual(ecg.dtype, np.float32)
            self.assertTrue(np.isfinite(ppg).all())
            np.testing.assert_allclose(ppg.min(axis=1), -1)
            np.testing.assert_allclose(ecg.max(axis=1), 1)
            # Offset recordings of the same wave should agree after alignment.
            self.assertGreater(np.corrcoef(ppg[1], ecg[1])[0, 1], 0.98)
            splits = train.split_subjects(subjects, held_out_subject=44)
            player_sets = [set(subjects[indices]) for indices in splits]
            self.assertEqual(player_sets[2], {44})
            for first, second in ((0, 1), (0, 2), (1, 2)):
                self.assertFalse(player_sets[first] & player_sets[second])

    def test_gaps_are_not_interpolated_into_training_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_session(root, 42, gap=True)
            with self.assertRaisesRegex(ValueError, "No usable paired windows"):
                alyx_data.load_alyx_segments(root, sampling_rate=130)

    def test_dvc_only_checkout_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "players").mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "DVC"):
                alyx_data.load_alyx_segments(root, sampling_rate=130)

    def test_ssl_uses_six_real_transformations_with_valid_labels(self):
        wave = np.sin(np.arange(64) / 5).astype(np.float32)[None]
        dataset = train.TransformationRecognitionDataset(wave)
        self.assertEqual(len(dataset), 6)
        for idx in range(6):
            transformed, label = dataset[idx]
            self.assertEqual(transformed.shape, (1, 64))
            self.assertEqual(label.item(), idx)
            self.assertTrue(torch.isfinite(transformed).all())
            self.assertTrue(torch.equal(transformed, dataset[idx][0]))
        np.testing.assert_allclose(dataset[4][0][0].numpy(), -wave[0])

    def test_ssl_and_gan_training_step(self):
        torch.set_num_threads(1)
        cfg = train.ModelConfig(
            base_channels=2, gen_transformer_dim=4, gen_transformer_layers=1,
            disc_conv_channels=(4, 4, 4), disc_conv_kernels=(5, 3, 3),
            disc_model_dim=4, disc_transformer_layers=1, disc_ffn_dim=8,
            disc_max_tokens=65, max_tokens=16, disc_dropout=0,
            batch_size=2, epochs_pretrain=1, epochs_gan=1, n_critic=1, device="cpu",
        )
        wave = np.sin(np.arange(64) / 5).astype(np.float32)[None].repeat(2, axis=0)
        ssl = DataLoader(train.TransformationRecognitionDataset(wave), batch_size=2)
        pretrained = train.train_ssl_pretraining(train.ECGTransformerDiscriminator(cfg), ssl, ssl, cfg)
        discriminator = train.transfer_pretrained_discriminator_backbone(
            pretrained, train.ECGTransformerDiscriminator(cfg), freeze_backbone=True)
        loader = DataLoader(train.PairedPPGECGDataset(wave, wave), batch_size=2)
        generator, _, history = train.train_wgan_gp(
            train.PPGtoECGGenerator(cfg), discriminator, loader, loader, cfg)
        self.assertTrue(np.isfinite(history["g_loss"]).all())
        self.assertEqual(generator(torch.from_numpy(wave).unsqueeze(1)).shape, (2, 1, 64))


if __name__ == "__main__":
    unittest.main()
