"""Fast checks of imports, models, configuration, and planning; no training or downloads."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from core.configuration import load_config
from core.launcher import config_for_seed
from core import homogeneous_engine as homogeneous
from core import heterogeneous_engine as heterogeneous
from experiments.homogeneous import build_default_config as homogeneous_config
from experiments.heterogeneous import build_default_config as heterogeneous_config
from experiments.width_study import build_default_config as width_config


class SmokeTests(unittest.TestCase):
    def test_configuration_inheritance(self):
        first, second = homogeneous_config(), width_config()
        for key in ("output_dir", "experiment_version"):
            first.pop(key)
            second.pop(key)
        self.assertEqual(first, second)

    def test_configuration_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cycle.json"
            path.write_text('{"extends": "cycle.json"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Circular"):
                load_config(path)

    def test_seed_mapping(self):
        base = heterogeneous_config()
        seeded = config_for_seed(base, 43)
        self.assertEqual(seeded["channel_seed_base"], 2000)
        self.assertEqual(seeded["label_sharding"]["class_latent_seed"], 43)
        self.assertEqual(seeded["heterogeneous_model"]["assignment_seed"], 43)
        self.assertEqual(seeded["physical"], base["physical"])
        self.assertEqual(base["training_seed"], 42)

    def test_model_shapes_and_parameter_counts(self):
        with torch.random.fork_rng(devices=[]):
            for architecture in ("teacher", "student"):
                model = homogeneous.get_model(architecture, 10).cpu().eval()
                with torch.no_grad():
                    logits = model(torch.zeros(2, 3, 32, 32))
                self.assertEqual(tuple(logits.shape), (2, 10))
                self.assertEqual(
                    sum(p.numel() for p in model.parameters()),
                    homogeneous.model_param_count_formula(architecture, 10),
                )

    def test_surrogate(self):
        value = homogeneous.evaluate_objective("heuristic_H1", 40, 39)
        self.assertAlmostEqual(value, 1.0 / 40 + 1.0 / 39)

    def test_homogeneous_planning(self):
        config = homogeneous_config()
        homogeneous.set_simulation_context(config)
        width, rounds, scan = homogeneous.run_stage1(128, "heuristic_H1", verbose=False)
        self.assertEqual([point["n_s"] for point in scan], config["stage1_candidate_widths"])
        self.assertEqual((width, rounds), (40, 39))

    def test_heterogeneous_planning(self):
        config = heterogeneous_config()
        heterogeneous.set_simulation_context(config)
        teachers = heterogeneous.get_fedkd_teacher_specs(config, 10)
        widths = [teacher["model_width"] for teacher in teachers]
        self.assertEqual([widths.count(w) for w in (72, 128, 184)], [3, 4, 3])
        width, rounds, _ = heterogeneous.run_stage1_with_heterogeneous_teachers(
            128, teachers, "heuristic_H1", verbose=False
        )
        self.assertEqual((width, rounds), (30, 34))

    def test_channel_reproducibility(self):
        homogeneous.set_simulation_context(homogeneous_config())
        first = homogeneous.generate_channel_sequence(2, seed_base=1000)
        second = homogeneous.generate_channel_sequence(2, seed_base=1000)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(np.asarray(first).shape, (2, 10, 100))


if __name__ == "__main__":
    unittest.main()
