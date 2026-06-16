"""
utils/checkpoint.py
Atomic checkpointing for Kaggle / cloud session-disconnect resilience.

Design
------
Every save writes to a temp file first, then os.replace() (POSIX atomic
rename) so a disconnection mid-write never corrupts the saved file.

A small JSON "manifest" records which (seed, method, rate, stage) have
already completed.  On resume, run_single_seed() skips completed stages
instead of recomputing them, which on 2×T4 can save hours.

Usage
-----
    ckpt = AtomicCheckpointer(output_dir, exp_name)
    ckpt.save_model(model, "base", seed=42)
    model = ckpt.load_model(arch, num_classes, device, "base", seed=42)
    ckpt.mark_done("base_finetune", seed=42)
    ckpt.is_done("base_finetune", seed=42)   # → True
"""

from __future__ import annotations
import json
import logging
import os
import tempfile
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AtomicCheckpointer:
    """
    Atomic filesystem checkpointer for Kaggle 2×T4 sessions.

    All saves are: write tmp → os.replace (kernel-level atomic rename).
    A JSON manifest tracks completed stages so restarts skip done work.
    """

    def __init__(self, output_dir: str, exp_name: str) -> None:
        self.ckpt_dir = os.path.join(output_dir, exp_name, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self._manifest_path = os.path.join(self.ckpt_dir, "manifest.json")
        self._manifest: dict = self._load_manifest()

    # ── manifest ──────────────────────────────────────────────────────────────

    def _load_manifest(self) -> dict:
        if os.path.exists(self._manifest_path):
            try:
                with open(self._manifest_path) as f:
                    m = json.load(f)
                logger.info(f"Loaded checkpoint manifest ({len(m)} entries)")
                return m
            except Exception as e:
                logger.warning(f"Could not read manifest ({e}), starting fresh")
        return {}

    def _save_manifest(self) -> None:
        self._atomic_json_write(self._manifest_path, self._manifest)

    def _stage_key(self, stage: str, seed: Optional[int] = None,
                   method: Optional[str] = None,
                   rate: Optional[float] = None) -> str:
        parts = [stage]
        if seed is not None:
            parts.append(f"s{seed}")
        if method is not None:
            parts.append(method)
        if rate is not None:
            parts.append(f"r{int(rate)}")
        return ":".join(parts)

    def mark_done(self, stage: str, seed: Optional[int] = None,
                  method: Optional[str] = None,
                  rate: Optional[float] = None) -> None:
        key = self._stage_key(stage, seed, method, rate)
        self._manifest[key] = True
        self._save_manifest()
        logger.info(f"[Checkpoint] marked done: {key}")

    def is_done(self, stage: str, seed: Optional[int] = None,
                method: Optional[str] = None,
                rate: Optional[float] = None) -> bool:
        key = self._stage_key(stage, seed, method, rate)
        return bool(self._manifest.get(key, False))

    # ── atomic I/O helpers ────────────────────────────────────────────────────

    def _atomic_write(self, dest_path: str, data: bytes) -> None:
        """Write bytes atomically to dest_path."""
        dir_ = os.path.dirname(dest_path)
        with tempfile.NamedTemporaryFile(dir=dir_, delete=False,
                                         suffix=".tmp") as f:
            f.write(data)
            tmp = f.name
        try:
            os.replace(tmp, dest_path)
        except Exception:
            os.unlink(tmp)
            raise

    def _atomic_json_write(self, dest_path: str, obj: object) -> None:
        dir_ = os.path.dirname(dest_path)
        with tempfile.NamedTemporaryFile(dir=dir_, delete=False,
                                         mode="w", suffix=".tmp") as f:
            json.dump(obj, f, indent=2)
            tmp = f.name
        try:
            os.replace(tmp, dest_path)
        except Exception:
            os.unlink(tmp)
            raise

    # ── model save / load ─────────────────────────────────────────────────────

    def _model_path(self, tag: str, seed: Optional[int] = None,
                    method: Optional[str] = None,
                    rate: Optional[float] = None) -> str:
        parts = [tag]
        if seed is not None:
            parts.append(f"seed{seed}")
        if method is not None:
            parts.append(method)
        if rate is not None:
            parts.append(f"r{int(rate)}pct")
        return os.path.join(self.ckpt_dir, "_".join(parts) + ".pth")

    def save_model(self, model: nn.Module, tag: str,
                   seed: Optional[int] = None,
                   method: Optional[str] = None,
                   rate: Optional[float] = None,
                   extra: Optional[dict] = None) -> str:
        """
        Atomically save model state_dict (+ optional extra dict) to disk.
        Returns the path written.
        """
        path = self._model_path(tag, seed, method, rate)
        payload = {"state_dict": model.state_dict()}
        if extra:
            payload.update(extra)
        # Use torch.save to a BytesIO, then atomic-write
        import io
        buf = io.BytesIO()
        torch.save(payload, buf)
        self._atomic_write(path, buf.getvalue())
        logger.info(f"[Checkpoint] saved: {os.path.basename(path)}")
        return path

    def load_model(self, model: nn.Module, tag: str,
                   seed: Optional[int] = None,
                   method: Optional[str] = None,
                   rate: Optional[float] = None,
                   device: Optional[torch.device] = None) -> tuple[nn.Module, dict]:
        """
        Load state_dict into model in-place.
        Returns (model, extra_dict).
        """
        path = self._model_path(tag, seed, method, rate)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        payload = torch.load(path, map_location=device or "cpu",
                             weights_only=False)
        model.load_state_dict(payload.pop("state_dict"))
        logger.info(f"[Checkpoint] loaded: {os.path.basename(path)}")
        return model, payload

    def checkpoint_exists(self, tag: str, seed: Optional[int] = None,
                          method: Optional[str] = None,
                          rate: Optional[float] = None) -> bool:
        return os.path.exists(self._model_path(tag, seed, method, rate))

    # ── result dict save / load ────────────────────────────────────────────────

    def save_result(self, result: dict, tag: str,
                    seed: Optional[int] = None) -> None:
        parts = [tag]
        if seed is not None:
            parts.append(f"seed{seed}")
        path = os.path.join(self.ckpt_dir, "_".join(parts) + ".json")
        self._atomic_json_write(path, result)

    def load_result(self, tag: str,
                    seed: Optional[int] = None) -> Optional[dict]:
        parts = [tag]
        if seed is not None:
            parts.append(f"seed{seed}")
        path = os.path.join(self.ckpt_dir, "_".join(parts) + ".json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)
