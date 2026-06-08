"""Examples-side glue shared by phase-N CIFAR-10 scripts.

This subpackage is the *instantiation layer* for the FWS CIFAR phases —
the WideKernelCNN-SiLU topology, the CIFAR-10 loader, the three baseline
arms that compose the ``fws-bench`` machinery with this specific
mainnet, the mainnet-specific α / G_leaf diagnostic helpers, and the
plots + research-log writer. The library-side machinery has moved
upstream:

- Stage-0 falsifier + N-arm paired training + ``Arm`` value type —
  :mod:`fws_bench`.
- Spectrum-fit power-law diagnostics (HT-SR α, radial-FFT α) and
  Jacobian / Hessian σ-spectra — :mod:`landscape_archaeology`.

Each phase file imports from all three: :mod:`fws_bench` for the
harness, :mod:`landscape_archaeology` for spectral probes, and
``_common`` for the CIFAR-specific instantiations and plots.

Submodules:

- :mod:`mainnet`     — WideKernelCNN-SiLU topology, leaf layout, direct
                        CNN forward + loss, ``init_cnn_params`` (W matched),
                        ``OverparamCNN`` (W overparam).
- :mod:`data`        — CIFAR-10 loader with ``~/.cache/cifar10``.
- :mod:`arms`        — Three baseline arm factories (``make_w_matched``,
                        ``make_w_overparam``, ``make_fws_parallel``) and an
                        FWS-hyper helper (``make_fws_hyper``). All return
                        :class:`fws_bench.Arm`. Owns the ``G_leaf``
                        primitive, ``ParallelGLeaves``, ``LEAF_COORDS``,
                        and the Wong palette.
- :mod:`diagnostics` — ``per_leaf_alphas`` (calls landscape-archaeology),
                        ``g_leaf_cosine_matrix``, ``count_params``,
                        ``global_l2_norm``. Mainnet-specific glue only.
- :mod:`reporting`   — Wong-palette plots, research-log writer, ``quill``
                        PDF render.

Phase scripts import these as a flat sibling package after prepending
``examples/`` to ``sys.path``::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from _common import arms, data, diagnostics, mainnet, reporting
"""

from __future__ import annotations

from . import arms, data, diagnostics, mainnet, reporting


__all__ = ["arms", "data", "diagnostics", "mainnet", "reporting"]
