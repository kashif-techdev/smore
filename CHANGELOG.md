# Changelog

## [1.1.0] - 2026-06-02

### Added

- Subject-level train/val/test splits (`src/smore/splits.py`) to prevent patient leakage
- Differentiable `CombinedLoss` (MSE + SSIM) in `src/smore/losses.py`
- CI workflow with smoke tests (`.github/workflows/ci.yml`)
- Workflow to set GitHub repository description and topics (`.github/workflows/repo-metadata.yml`)
- `CITATION.cff` for software citation metadata
- `scripts/smoke_test.py` and `scripts/set_github_about.py`

### Changed

- Notebook uses `subject_level_split` and `CombinedLoss` (v2 training)
- README updated for v1.1 features and project layout

## [1.0.0] - 2026-06-02

### Added

- Initial SMORE DIP Jupyter pipeline, documentation assets, and README
