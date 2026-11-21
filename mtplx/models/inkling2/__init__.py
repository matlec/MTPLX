# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 (``inkling2`` / ``inkling2_loop``) text model for MTPLX.

Ported from oMLX v0.8.2; see the per-file headers.
"""

from __future__ import annotations

# Looped checkpoints have their own model_type and the same model.
INKLING2_MODEL_TYPES = frozenset({"inkling2", "inkling2_loop"})

__all__ = ["INKLING2_MODEL_TYPES"]
