"""Themed fixtures shared by phase-N CIFAR-10 example scripts.

Phases 8/9/10 of the FWS programme share ~95% of their code (WideKernelCNN
mainnet, CIFAR-10 loader, W-matched / W-overparam / FWS-parallel-no-G_H
baselines, Adam(1e-3) paired training, HT-SR / radial-FFT / Jacobian-σ /
Hessian diagnostics, the same eight plots, and the research-log writer).
The actual phase-specific architectural change is the FWS-hyper arm's
``G_H`` and renderer, ~50-100 LoC per phase.

This subpackage owns the shared 95%; each phase file owns its own
``G_H`` + ``fws_hyper`` arm + ``main()`` wiring.

Submodules:

- :mod:`mainnet`    — WideKernelCNN-SiLU topology, leaf layout, direct CNN forward.
- :mod:`data`       — CIFAR-10 loader with ``~/.cache/cifar10``.
- :mod:`arms`       — ``Arm`` value type + ``ParallelGLeaves`` + the three
                       common arms (W matched, W overparam, FWS-parallel-no-G_H).
- :mod:`training`   — paired step + Stage-0 ``σ_min`` falsifier orchestration.
- :mod:`diagnostics`— HT-SR α, radial-FFT α, σ-spectrum, Hessian top-eig probes.
- :mod:`reporting`  — Wong-palette plots, research-log writer, ``quill`` PDF render.

Phase scripts import these as a flat sibling package after prepending
``examples/`` to ``sys.path``::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from _common import mainnet, data, arms, training, diagnostics, reporting
"""

from __future__ import annotations

from . import arms, data, diagnostics, mainnet, reporting, training


__all__ = ["arms", "data", "diagnostics", "mainnet", "reporting", "training"]
