# Repo Structure Plan (Non-Destructive)

This plan describes a cleaner structure without deleting any essential research or paper-trading components.

## Proposed Top-Level Layout

1. `docs/` for PRD, phase docs, governance, and checklists.
2. `python/` for the authoritative research, validation, and paper-trading pipeline.
3. `cpp/` for the optional C++ core (archived unless reactivated).
4. `artifacts/` for generated outputs that are not source code.
5. `data/` for raw and derived datasets (already excluded from git).

## Mapping From Current Layout

1. `AlphaMomentum_PRD_v4.md`, `01_phase1_infrastructure_and_data.md`, `03_broker_readiness_and_live_execution.md`, `04_phase4_forward_monitoring_and_governance.md`, and `WINDOWS_SETUP_CHECKLIST.md` -> `docs/`
2. `results/`, `portfolio/`, `papertrading/`, `_snapshots/` -> `artifacts/`
3. Keep `python/` and `cpp/` in place

## Safety Rules

1. Do not delete any research or validation outputs needed for auditability.
2. Move folders only after confirming that all scripts accept a configurable output root.
3. Preserve the frozen Phase 4 manifests and the employer-facing package.
