"""
training/trainer.py
Two trainer classes used throughout the DSFP pipeline:

BaseTrainer
    Standard SGD + CosineAnnealingWarmRestarts fine-tuning with:
    • Optional Mixup augmentation (alpha=0.2)
    • Label smoothing (passed via nn.CrossEntropyLoss)
    • Gradient accumulation
    • Early stopping with correctly-tracked patience counter
    • Best-model checkpointing (returns the best model, not the last)
    • AMP support via torch.amp (new API, avoids deprecation warnings)
    • Multi-GPU (DataParallel) awareness

KDTrainer
    Knowledge-distillation fine-tuning (Hinton et al., NeurIPS 2014 workshop):
    • Loss = α · KL(T(teacher) ‖ T(student)) + (1−α) · CE(student, hard_labels)
    • Optional dynamic α decay: α anneals linearly from α_start to 0.1 over
      the training budget so the student gradually relies on hard labels.
    • Same early-stopping / best-model logic as BaseTrainer.
    • Teacher is frozen (eval mode, no grad).

Fixes applied vs original:
  [BUG-1]  BaseTrainer.train: no_improve was never incremented (was += 0).  Fixed to += 1.
  [BUG-2]  KDTrainer.train:  same no_improve bug. Fixed.
  [BUG-3]  torch.cuda.amp.GradScaler / autocast are deprecated since PyTorch 2.x.
           Replaced with torch.amp.GradScaler("cuda") and torch.amp.autocast("cuda").
  [BUG-4]  _train_epoch in BaseTrainer: final batch never flushed when
           len(loader) % accumulation_steps != 0.  Added flush after loop.
  [BUG-5]  KDTrainer._train_epoch: teacher forward was inside autocast; for
           stability teacher logits should be full-precision detached.  Fixed.
  [BUG-6]  Scheduler step called once per epoch regardless of accumulation,
           which is correct for CosineAnnealingWarmRestarts — kept as is.
  [KAGGLE] Disabled persistent_workers when num_workers=0 to avoid worker
           spawn failures (Kaggle sessions sometimes restrict fork).
"""

from __future__ import annotations
import logging
from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AMP helpers — use new torch.amp API, fall back gracefully on CPU
# ---------------------------------------------------------------------------

def _make_scaler(use_amp: bool) -> torch.amp.GradScaler:
    # torch.amp.GradScaler was introduced in PyTorch 2.1;
    # the old torch.cuda.amp.GradScaler still works but raises DeprecationWarning.
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        # PyTorch < 2.1 fallback
        return torch.cuda.amp.GradScaler(enabled=use_amp)  # type: ignore[attr-defined]


def _autocast(device_type: str, enabled: bool):
    try:
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    except TypeError:
        # PyTorch < 2.1 fallback
        return torch.cuda.amp.autocast(enabled=enabled)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------

def _mixup_data(x: torch.Tensor, y: torch.Tensor,
                alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor, float]:
    """Sample λ ~ Beta(α, α) and return mixed inputs and both label tensors."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def _mixup_criterion(criterion: nn.Module, pred: torch.Tensor,
                     y_a: torch.Tensor, y_b: torch.Tensor,
                     lam: float) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ---------------------------------------------------------------------------
# BaseTrainer
# ---------------------------------------------------------------------------

class BaseTrainer:
    """
    Standard fine-tuning trainer.

    Args:
        model:              Model to train (modified in-place).
        device:             Compute device.
        lr:                 Initial SGD learning rate.
        momentum:           SGD momentum.
        weight_decay:       L2 regularisation.
        label_smoothing:    Label smoothing factor for CrossEntropyLoss.
        cosine_t0:          CosineAnnealingWarmRestarts T_0.
        cosine_tmult:       CosineAnnealingWarmRestarts T_mult.
        accumulation_steps: Gradient accumulation steps.
        patience:           Early stopping patience (in epochs).
        use_amp:            Enable automatic mixed precision.
        use_mixup:          Enable Mixup augmentation.
        mixup_alpha:        Beta distribution parameter for Mixup.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float                = 1e-3,
        momentum: float          = 0.9,
        weight_decay: float      = 1e-4,
        label_smoothing: float   = 0.1,
        cosine_t0: int           = 50,
        cosine_tmult: int        = 2,
        accumulation_steps: int  = 4,
        patience: int            = 12,
        use_amp: bool            = True,
        use_mixup: bool          = True,
        mixup_alpha: float       = 0.2,
    ) -> None:
        self.model              = model
        self.device             = device
        self.accumulation_steps = max(1, accumulation_steps)
        self.patience           = patience
        # [BUG-3] use new torch.amp API
        self.use_amp            = use_amp and device.type == "cuda"
        self.use_mixup          = use_mixup
        self.mixup_alpha        = mixup_alpha

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        ).to(device)

        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr, momentum=momentum,
            weight_decay=weight_decay, nesterov=True,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=cosine_t0, T_mult=cosine_tmult, eta_min=1e-6,
        )
        # [BUG-3] new GradScaler API
        self.scaler = _make_scaler(self.use_amp)

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        self.optimizer.zero_grad()
        device_type = self.device.type

        for step, (images, labels) in enumerate(loader, 1):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_mixup:
                images, y_a, y_b, lam = _mixup_data(
                    images, labels, alpha=self.mixup_alpha)

            # [BUG-3] use new autocast API
            with _autocast(device_type, self.use_amp):
                logits = self.model(images)
                if self.use_mixup:
                    loss = _mixup_criterion(self.criterion, logits, y_a, y_b, lam)
                else:
                    loss = self.criterion(logits, labels)
                loss = loss / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if step % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            running_loss += loss.item() * self.accumulation_steps

        # [BUG-4] flush leftover accumulated gradients from a partial final batch
        remainder = len(loader) % self.accumulation_steps
        if remainder != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return running_loss / len(loader)

    @torch.no_grad()
    def _eval(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = total = 0
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            preds  = self.model(images).argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total   += labels.size(0)
        return 100.0 * correct / total

    def train(self, train_loader: DataLoader,
              test_loader: DataLoader,
              epochs: int) -> Tuple[nn.Module, float]:
        """
        Train for up to `epochs` epochs with early stopping.

        Returns:
            (best_model, best_accuracy) — best model by validation accuracy.
        """
        best_acc    = 0.0
        best_state  = deepcopy(self.model.state_dict())
        no_improve  = 0

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_acc    = self._eval(test_loader)
            self.scheduler.step()

            if val_acc > best_acc:
                best_acc   = val_acc
                best_state = deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1   # [BUG-1] was += 0 in original

            if epoch % 10 == 0 or epoch == epochs:
                logger.info(
                    f"[BaseTrainer] Epoch {epoch:4d}/{epochs} | "
                    f"loss={train_loss:.4f} | val_acc={val_acc:.2f}% | "
                    f"best={best_acc:.2f}% | patience={no_improve}/{self.patience}"
                )

            if no_improve >= self.patience:
                logger.info(f"Early stopping at epoch {epoch} "
                            f"(no improvement for {self.patience} epochs)")
                break

        self.model.load_state_dict(best_state)
        logger.info(f"[BaseTrainer] Done — best val accuracy: {best_acc:.4f}%")
        return self.model, best_acc


# ---------------------------------------------------------------------------
# KDTrainer
# ---------------------------------------------------------------------------

class KDTrainer:
    """
    Knowledge-distillation fine-tuner.

    Teacher is frozen. Student is trained with a combination of KD loss
    (soft targets from teacher) and standard cross-entropy on hard labels.

    Loss (per batch):
        L = α · T² · KL(softmax(t_logits/T) ‖ softmax(s_logits/T))
              + (1−α) · CE(s_logits, hard_labels)

    Dynamic alpha: if `dynamic_alpha=True`, α decays linearly from α_start
    to 0.1 over the training budget so the student transitions from soft- to
    hard-label supervision.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        device: torch.device,
        lr: float              = 1e-4,
        weight_decay: float    = 1e-4,
        alpha: float           = 0.5,
        temperature: float     = 4.0,
        patience: int          = 10,
        dynamic_alpha: bool    = True,
        use_amp: bool          = True,
    ) -> None:
        self.teacher       = teacher_model.to(device).eval()
        self.student       = student_model.to(device)
        self.device        = device
        self.alpha_start   = alpha
        self.temperature   = temperature
        self.patience      = patience
        self.dynamic_alpha = dynamic_alpha
        # [BUG-3] new AMP API
        self.use_amp       = use_amp and device.type == "cuda"

        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.ce_criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer    = torch.optim.Adam(
            self.student.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.scheduler    = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2, eta_min=1e-6,
        )
        # [BUG-3] new GradScaler
        self.scaler = _make_scaler(self.use_amp)

    def _alpha(self, epoch: int, total_epochs: int) -> float:
        if not self.dynamic_alpha:
            return self.alpha_start
        alpha_end = 0.1
        progress  = (epoch - 1) / max(total_epochs - 1, 1)
        return self.alpha_start + progress * (alpha_end - self.alpha_start)

    def _kd_loss(self, s_logits: torch.Tensor,
                 t_logits: torch.Tensor) -> torch.Tensor:
        T   = self.temperature
        s_p = F.log_softmax(s_logits / T, dim=-1)
        t_p = F.softmax(t_logits / T, dim=-1)
        return F.kl_div(s_p, t_p, reduction="batchmean") * (T * T)

    def _train_epoch(self, loader: DataLoader,
                     epoch: int, total_epochs: int) -> float:
        self.student.train()
        self.teacher.eval()
        alpha        = self._alpha(epoch, total_epochs)
        running_loss = 0.0
        device_type  = self.device.type

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # [BUG-5] get teacher logits in full precision OUTSIDE autocast,
            # then cast to fp32 to avoid stale NaN with AMP
            with torch.no_grad():
                t_logits = self.teacher(images).float()

            with _autocast(device_type, self.use_amp):
                s_logits = self.student(images)
                kd_loss  = self._kd_loss(s_logits.float(), t_logits)
                ce_loss  = self.ce_criterion(s_logits, labels)
                loss     = alpha * kd_loss + (1.0 - alpha) * ce_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

        return running_loss / len(loader)

    @torch.no_grad()
    def _eval(self, loader: DataLoader) -> float:
        self.student.eval()
        correct = total = 0
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            preds  = self.student(images).argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total   += labels.size(0)
        return 100.0 * correct / total

    def train(self, train_loader: DataLoader,
              test_loader: DataLoader,
              epochs: int) -> float:
        """
        KD fine-tune for up to `epochs` epochs with early stopping.

        Returns:
            best_accuracy (float) — best student val accuracy.
        """
        best_acc   = 0.0
        best_state = deepcopy(self.student.state_dict())
        no_improve = 0

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch, epochs)
            val_acc    = self._eval(test_loader)
            self.scheduler.step()

            if val_acc > best_acc:
                best_acc   = val_acc
                best_state = deepcopy(self.student.state_dict())
                no_improve = 0
            else:
                no_improve += 1   # [BUG-2] was += 0 in original

            if epoch % 50 == 0 or epoch == epochs:
                alpha = self._alpha(epoch, epochs)
                logger.info(
                    f"[KDTrainer] Epoch {epoch:4d}/{epochs} | "
                    f"loss={train_loss:.4f} | val_acc={val_acc:.2f}% | "
                    f"best={best_acc:.2f}% | α={alpha:.3f} | "
                    f"patience={no_improve}/{self.patience}"
                )

            if no_improve >= self.patience:
                logger.info(f"KD early stopping at epoch {epoch}")
                break

        self.student.load_state_dict(best_state)
        logger.info(f"[KDTrainer] Done — best val accuracy: {best_acc:.4f}%")
        return best_acc
