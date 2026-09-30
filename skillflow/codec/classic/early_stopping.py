"""Pure early-stopping state updates for codec training."""

from __future__ import annotations

import math
from typing import NamedTuple


class EarlyStoppingUpdate(NamedTuple):
    best_validation_loss: float
    best_epoch: int
    stale_epochs: int
    improved: bool
    should_stop: bool


def update_early_stopping(
    *,
    epoch: int,
    validation_loss: float,
    best_validation_loss: float,
    best_epoch: int,
    stale_epochs: int,
    track_best: bool,
    min_epochs: int,
    patience: int,
    min_delta: float = 1e-6,
) -> EarlyStoppingUpdate:
    """Update checkpoint and stopping state without mutating training objects.

    ``min_epochs`` controls only when training may stop.  Best-checkpoint
    updates and patience accounting begin as soon as ``track_best`` is true.
    """

    current_epoch = int(epoch)
    if current_epoch < 1:
        raise ValueError(f"epoch must be one-based and positive, observed={epoch}")

    finite = math.isfinite(float(validation_loss))
    improved = (
        bool(track_best)
        and finite
        and float(validation_loss) < float(best_validation_loss) - float(min_delta)
    )
    next_best_loss = float(best_validation_loss)
    next_best_epoch = int(best_epoch)
    next_stale_epochs = int(stale_epochs)
    if improved:
        next_best_loss = float(validation_loss)
        next_best_epoch = current_epoch
        next_stale_epochs = 0
    elif bool(track_best) and finite:
        next_stale_epochs += 1

    normalized_patience = max(0, int(patience))
    eligible_for_stop = bool(track_best) and current_epoch >= max(1, int(min_epochs))
    should_stop = (
        normalized_patience > 0
        and finite
        and eligible_for_stop
        and next_stale_epochs >= normalized_patience
    )
    return EarlyStoppingUpdate(
        best_validation_loss=next_best_loss,
        best_epoch=next_best_epoch,
        stale_epochs=next_stale_epochs,
        improved=improved,
        should_stop=should_stop,
    )
