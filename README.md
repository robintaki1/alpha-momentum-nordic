# Alpha Momentum (Nordic)

This repo contains the research, validation, and paper-trading pipeline for the Alpha Momentum Nordic system.
The authoritative process and gating live in the PRD and phase docs. Do not skip the gates.

Key docs:
- `C:\Users\robin\Desktop\alpha momentum\AlphaMomentum_PRD_v4.md`
- `C:\Users\robin\Desktop\alpha momentum\01_phase1_infrastructure_and_data.md`
- `C:\Users\robin\Desktop\alpha momentum\04_phase4_forward_monitoring_and_governance.md`
- `C:\Users\robin\Desktop\alpha momentum\03_broker_readiness_and_live_execution.md`
- `C:\Users\robin\Desktop\alpha momentum\WINDOWS_SETUP_CHECKLIST.md`

Status (per PRD):
- No authoritative validated winner is currently active.
- Preserved forward-monitor and paper-trading artifacts are reference-only until a future candidate revalidates.
- Live engine is not active and must not be started until a future candidate survives authoritative validation and later clears the paper-trading gate.

## Repository Map

Core Python pipeline:
- Phase 1 data + validation: `C:\Users\robin\Desktop\alpha momentum\python\download_eodhd.py`, `C:\Users\robin\Desktop\alpha momentum\python\download_riksbank_fx.py`, `C:\Users\robin\Desktop\alpha momentum\python\build_universe.py`, `C:\Users\robin\Desktop\alpha momentum\python\validate.py`
- Research + selection: `C:\Users\robin\Desktop\alpha momentum\python\research_engine.py`, `C:\Users\robin\Desktop\alpha momentum\python\grid_search.py`, `C:\Users\robin\Desktop\alpha momentum\python\walk_forward.py`
- Cadence comparison: `C:\Users\robin\Desktop\alpha momentum\python\cadence_engine.py`, `C:\Users\robin\Desktop\alpha momentum\python\run_cadence_compare.py`
- Forward monitor + paper trading: `C:\Users\robin\Desktop\alpha momentum\python\forward_monitor.py`, `C:\Users\robin\Desktop\alpha momentum\python\paper_trade_tracker.py`
- Automatic monthly refresh + rebalance entry point: `C:\Users\robin\Desktop\alpha momentum\python\monthly_rebalance_runner.py` via `C:\Users\robin\Desktop\alpha momentum\run_monthly_rebalance.bat`

Configuration:
- Python source of truth: `C:\Users\robin\Desktop\alpha momentum\config.py`
- C++ legacy config: `C:\Users\robin\Desktop\alpha momentum\config_cpp.json` (use only if reactivating C++)

Outputs and artifacts:
- Data artifacts (excluded from git): `C:\Users\robin\Desktop\alpha momentum\data`
- Research results + dashboards: `C:\Users\robin\Desktop\alpha momentum\results`
- Paper-trading logs: `C:\Users\robin\Desktop\alpha momentum\papertrading`
- Employer-facing package: `C:\Users\robin\Desktop\alpha momentum\portfolio`
- Snapshots: `C:\Users\robin\Desktop\alpha momentum\_snapshots`

## C++ Core Status

The C++ core remains in the repo (`C:\Users\robin\Desktop\alpha momentum\cpp`) but the paper-trading branch is
Python-first per the PRD. If C++ is reintroduced, confirm config parity with `C:\Users\robin\Desktop\alpha momentum\config.py`
and update the validation gate to enforce C++ == Python.
