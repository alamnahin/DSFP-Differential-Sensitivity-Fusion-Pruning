"""
training/trainer.py
Two trainer classes used throughout the DSFP pipeline:

BaseTrainer
    Standard SGD + CosineAnnealingWarmRestarts fine-tuning with:
    • Optional Mixup augmentation (alpha=0.2)
    • Label smoothing (passed via nn.CrossEntropyLoss)
    • Gradient accumulation
    • Early stopping with a correctly-tracked patience counter
    • Best-model checkpointing (returns the best model, not the last)
    • AMP support via torch.cuda.amp

KDTrainer
    Knowledge-distillation fine-tuning (Hinton et al., NeurIPS 2014 workshop):
    • Loss = α · KL(T(teacher) ‖ T(student)) + (1−α) · CE(student, hard_labels)
    • Optional dynamic α decay: α anneals linearly from α_start to 0.1 over
      the training budget so the student gradually relies on hard labels.
    • Same early-stopping / best-model logic as BaseTrainer.
    • Teacher is frozen (eval mode, no grad).
"""

from __future__ import annotations
import logging
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------

def _mixup_data(x: torch.Tensor, y: torch.Tensor,
                alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor, float]:
    """Sample λ ~ Beta(α, α) and return mixed inputs and both label tensors."""
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def _mixup_criterion(criterion: nn.Module,
                     pred: torch.Tensor,
                     y_a: torch.Tensor,
                     y_b: torch.Tensor,
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
        self.accumulation_steps = accumulation_steps
        self.patience           = patience
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
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        self.optimizer.zero_grad()

        for step, (images, labels) in enumerate(loader, 1):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_mixup:
                images, y_a, y_b, lam = _mixup_data(
                    images, labels, alpha=self.mixup_alpha)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
                if self.use_mixup:
                    loss = _mixup_criterion(
                        self.criterion, logits, y_a, y_b, lam)
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
            (best_model, best_accuracy)   — best model by validation accuracy.
        """
        best_acc    = 0.0
        best_state  = deepcopy(self.model.state_dict())
        no_improve  = 0   # correctly incremented early-stopping counter

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_acc    = self._eval(test_loader)
            self.scheduler.step()

            if val_acc > best_acc:
                best_acc   = val_acc
                best_state = deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1   # ← fixed: was += 0 in original code

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

        # Restore best weights before returning
        self.model.load_state_dict(best_state)
        logger.info(f"[BaseTrainer] Done — best val accuracy: {best_acc:.4f}%")
        return self.model, best_acc


# ---------------------------------------------------------------------------
# KDTrainer
# ---------------------------------------------------------------------------

class KDTrainer:
    """
    Knowledge-distillation fine-tuner.

    Teacher is frozen.  Student is trained with a combination of KD loss
    (soft targets from teacher) and standard cross-entropy on hard labels.

    Loss (per batch):
        L = α · T² · KL(softmax(t_logits/T) ‖ softmax(s_logits/T))
              + (1−α) · CE(s_logits, hard_labels)

    Dynamic alpha: if `dynamic_alpha=True`, α decays linearly from α_start
    to 0.1 over the training budget so the student transitions from soft- to
    hard-label supervision.

    Args:
        teacher_model:   Frozen reference model.
        student_model:   Model being fine-tuned (modified in-place).
        device:          Compute device.
        lr:              Adam learning rate.
        weight_decay:    L2 regularisation.
        alpha:           Initial KD loss weight (0 = CE only, 1 = KD only).
        temperature:     Softmax temperature T for soft targets.
        patience:        Early stopping patience.
        dynamic_alpha:   Linearly anneal α to 0.1.
        use_amp:         Enable AMP.
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
        self.use_amp       = use_amp and device.type == "cuda"

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.ce_criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer    = torch.optim.Adam(
            self.student.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.scheduler    = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2, eta_min=1e-6,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _alpha(self, epoch: int, total_epochs: int) -> float:
        """Return current alpha value (constant or linearly decayed)."""
        if not self.dynamic_alpha:
            return self.alpha_start
        # Linear decay: alpha_start → 0.1 over total_epochs
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

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                with torch.no_grad():
                    t_logits = self.teacher(images)
                s_logits = self.student(images)

                kd_loss = self._kd_loss(s_logits, t_logits)
                ce_loss = self.ce_criterion(s_logits, labels)
                loss    = alpha * kd_loss + (1.0 - alpha) * ce_loss

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
            best_accuracy (float)  — best student val accuracy.
        """
        best_acc   = 0.0
        best_state = deepcopy(self.student.state_dict())
        no_improve = 0   # correctly incremented early-stopping counter

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch, epochs)
            val_acc    = self._eval(test_loader)
            self.scheduler.step()

            if val_acc > best_acc:
                best_acc   = val_acc
                best_state = deepcopy(self.student.state_dict())
                no_improve = 0
            else:
                no_improve += 1   # ← fixed: was += 0 in original code

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
