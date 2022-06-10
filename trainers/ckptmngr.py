from pathlib import Path
from typing import Callable, List

import torch


class CheckpointManager:
    def __init__(
        self,
        ckptdir: Path,
        interval: int,
        compare_fn: Callable[[float, float], float],
        elements: List,
    ):
        self.ckptdir = ckptdir
        self.interval = interval
        self.compare_fn = compare_fn
        self.elements = elements

        self.ckptdir.mkdir(parents=True, exist_ok=True)
        self.initialized = False
        self.best_score = 0.0

    def save_interval(self, epoch_num: int):
        if epoch_num % self.interval == self.interval - 1:
            for ckpt_path in self.ckptdir.glob("*.pt"):
                if "best" not in ckpt_path.name:
                    ckpt_path.unlink()
            self._save_ckpt(str(epoch_num + 1), epoch_num + 1)

    def save_best(self, epoch_num: int, epoch_score: float):
        if not self.initialized:
            save = True
            self.initialized = True
        else:
            save = self.compare_fn(self.best_score, epoch_score) < 0
        print(f"#### best score {self.best_score:.2f} epoch score {epoch_score:.2f}", end=" ")
        if save:
            print("-> SAVED")
            for ckpt_path in self.ckptdir.glob("*.pt"):
                if "best" in ckpt_path.name:
                    ckpt_path.unlink()
            self._save_ckpt("best_" + str(epoch_num + 1), epoch_num + 1)
            self.best_score = epoch_score
        else:
            print("-> NOT SAVED")

    def try_to_load(self, best: bool):
        for ckpt_path in self.ckptdir.glob("*.pt"):
            if (best and "best" in ckpt_path.name) or (not best and "best" not in ckpt_path.name):
                return self._load_ckpt(ckpt_path.stem)
        return 0

    def _save_ckpt(self, ckpt_name: str, epoch_num: int):
        ckpt = {}
        ckpt["epoch"] = epoch_num
        ckpt["best_score"] = self.best_score
        for i in range(len(self.elements)):
            ckpt[str(i)] = self.elements[i].state_dict()

        torch.save(ckpt, self.ckptdir / (ckpt_name + ".pt"))

    def _load_ckpt(self, ckpt_name: str):
        ckpt = torch.load(self.ckptdir / (ckpt_name + ".pt"))
        epoch_num = ckpt["epoch"]
        self.best_score = ckpt["best_score"]
        self.initialized = True

        for i in range(len(self.elements)):
            self.elements[i].load_state_dict(ckpt[str(i)])

        return epoch_num
