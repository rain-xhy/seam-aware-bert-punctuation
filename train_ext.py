"""train.py with two optional extras, without editing train.py:

  --label_scheme {8,3,2}     train on coarser labels (punct.label_scheme)
  --pos_offset_prob P        random position offset in a fraction P of steps

Every other argument is passed to train.py unchanged, e.g.

  python train_ext.py --label_scheme 3 --data_dir prepared_data_complete \
      --out_dir checkpoints/T6_label3 --bert_dir bert-base-chinese --epochs 3 \
      --batch_size 1 --grad_accum 8 --max_len 400 --seed 42

Resume (resume.pt) works as in train.py. --label_scheme with --class_weights
is rejected: train.py computes the class weights from the 8-class labels.
"""
import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import train as base  # noqa: E402
from punct.label_scheme import coarsen  # noqa: E402
from punct.pos_offset import install_position_offset  # noqa: E402


class MappedLoader:
    """Wraps a DataLoader and coarsens the labels of every batch."""

    def __init__(self, loader, scheme):
        self.loader, self.scheme = loader, scheme

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            batch = dict(batch)
            batch["labels"] = coarsen(batch["labels"], self.scheme)
            yield batch


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--label_scheme", default="8", choices=["8", "3", "2"])
    ap.add_argument("--pos_offset_prob", type=float, default=0.0)
    ext, rest = ap.parse_known_args()
    if ext.label_scheme != "8" and "--class_weights" in rest:
        raise SystemExit("--label_scheme other than 8 cannot be combined with --class_weights")

    orig_train, orig_validate = base.train_one_epoch, base.validate
    state = {"hook": None}

    def train_one_epoch(model, dataloader, *a, **kw):
        if state["hook"] is None and ext.pos_offset_prob > 0:
            state["hook"] = install_position_offset(model, ext.pos_offset_prob)
        if ext.label_scheme != "8":
            dataloader = MappedLoader(dataloader, ext.label_scheme)
        return orig_train(model, dataloader, *a, **kw)

    def validate(model, dataloader, device):
        if ext.label_scheme != "8":
            dataloader = MappedLoader(dataloader, ext.label_scheme)
        return orig_validate(model, dataloader, device)

    base.train_one_epoch, base.validate = train_one_epoch, validate
    sys.argv = [sys.argv[0]] + rest
    base.main()


if __name__ == "__main__":
    main()
