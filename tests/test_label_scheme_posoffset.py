import unittest

import numpy as np
import torch
from transformers import BertConfig, BertForTokenClassification

from punct.label_scheme import coarsen
from punct.pos_offset import install_position_offset


class LabelSchemeTests(unittest.TestCase):
    def test_numpy_and_ignore(self):
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, -100])
        self.assertEqual(coarsen(x, "3").tolist(), [0, 1, 2, 2, 2, 1, 1, 1, -100])
        self.assertEqual(coarsen(x, "2").tolist(), [0, 1, 1, 1, 1, 1, 1, 1, -100])
        self.assertEqual(coarsen(x, "8").tolist(), x.tolist())

    def test_torch_matches_numpy(self):
        x = np.array([[0, 2, 7, -100], [4, 3, 1, 0]])
        for s in ("8", "3", "2"):
            np.testing.assert_array_equal(coarsen(torch.from_numpy(x), s).numpy(), coarsen(x, s))

    def test_unknown_scheme(self):
        with self.assertRaises(ValueError):
            coarsen(np.zeros(3, dtype=int), "5")


class PositionOffsetTests(unittest.TestCase):
    def setUp(self):
        cfg = BertConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=32, max_position_embeddings=64, num_labels=8)
        self.model = BertForTokenClassification(cfg)
        self.ids = torch.randint(1, 50, (2, 10))

    def test_changes_output_in_training_only(self):
        self.model.eval()
        with torch.no_grad():
            base_eval = self.model(self.ids).logits
        install_position_offset(self.model, prob=1.0, max_positions=64)
        with torch.no_grad():
            self.assertTrue(torch.equal(base_eval, self.model(self.ids).logits))  # eval untouched
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
        outs = [self.model(self.ids).logits.detach() for _ in range(8)]
        self.assertTrue(any(not torch.equal(outs[0], o) for o in outs[1:]))

    def test_prob_zero_installs_nothing(self):
        self.assertIsNone(install_position_offset(self.model, prob=0.0))

    def test_sequence_filling_all_positions_is_left_alone(self):
        install_position_offset(self.model, prob=1.0, max_positions=10)
        self.model.train()
        self.model(self.ids)  # seq == max_positions: no room, must not raise


if __name__ == "__main__":
    unittest.main()
