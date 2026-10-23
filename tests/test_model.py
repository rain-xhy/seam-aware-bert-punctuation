"""Tests for punct.model. Needs the real bert-base-chinese files and a GPU-free
CPU forward pass; no training happens here."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from punct.model import NUM_LABELS, encoder_weights_changed, get_weight_check_tensor, load_model, save_model

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bert-base-chinese")


@unittest.skipUnless(os.path.exists(MODEL_DIR), "bert-base-chinese/ not present")
class TestLoadModel(unittest.TestCase):
    def test_loads_with_correct_num_labels(self):
        model = load_model(MODEL_DIR)
        self.assertEqual(model.config.num_labels, NUM_LABELS)

    def test_forward_pass_shapes(self):
        model = load_model(MODEL_DIR)
        model.eval()
        input_ids = torch.randint(1000, 20000, (2, 50))
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        self.assertEqual(out.logits.shape, (2, 50, NUM_LABELS))

    def test_save_and_reload_round_trip(self):
        model = load_model(MODEL_DIR)
        with tempfile.TemporaryDirectory() as tmp:
            save_model(model, tmp)
            reloaded = load_model(tmp)
        self.assertEqual(reloaded.config.num_labels, NUM_LABELS)


@unittest.skipUnless(os.path.exists(MODEL_DIR), "bert-base-chinese/ not present")
class TestEncoderWeightCheck(unittest.TestCase):
    def test_unchanged_weights_detected_as_unchanged(self):
        model = load_model(MODEL_DIR)
        before = get_weight_check_tensor(model)
        after = get_weight_check_tensor(model)  # no training happened in between
        changed, max_diff = encoder_weights_changed(before, after)
        self.assertFalse(changed)
        self.assertEqual(max_diff, 0.0)

    def test_manually_perturbed_weights_detected_as_changed(self):
        model = load_model(MODEL_DIR)
        before = get_weight_check_tensor(model)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name == "bert.encoder.layer.0.attention.self.query.weight":
                    param.add_(0.01)
        after = get_weight_check_tensor(model)
        changed, max_diff = encoder_weights_changed(before, after)
        self.assertTrue(changed)
        self.assertGreater(max_diff, 0.0)

    def test_shape_mismatch_raises(self):
        before = torch.zeros(10, 10)
        after = torch.zeros(5, 5)
        with self.assertRaises(ValueError):
            encoder_weights_changed(before, after)

    def test_missing_param_path_raises(self):
        model = load_model(MODEL_DIR)
        with self.assertRaises(KeyError):
            get_weight_check_tensor(model, path="this.does.not.exist")


if __name__ == "__main__":
    unittest.main()
