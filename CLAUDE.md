# CLAUDE.md

@AGENTS.md

## Project Context

Experimental orchestration for the FWS programme. Consumes sibling libraries `loom` (re-parameterisation substrate), `ondes` (INR primitives), `jacobian-spec` (Jacobian-based spectral diagnostics), and `landscape-archaeology` (loss-landscape probes), and orchestrates paired z-space vs W-space training under matched compute. Public surface: `fws_bench.paired_train(...)` plus a `Regime` enum (`JOINT` / `ALTERNATING` / `META`). See `docs/PHILOSOPHY.md` for the paired-arm contract and the regime taxonomy.

## Scope Boundary

fws-bench owns the **experimental layer** — the harness, the regime switcher, the `optax.multi_transform` G/z partition setup, and the ablation runners. It does **not** own primitives that belong to a sibling:

- INR bodies, Fourier encodings → `ondes`
- The `render` re-parameterisation substrate → `loom`
- Jacobian / spectral diagnostics → `jacobian-spec`
- Loss-landscape probes → `landscape-archaeology`

If a feature naturally lives in a sibling, push it down. fws-bench composes; it does not re-implement.
