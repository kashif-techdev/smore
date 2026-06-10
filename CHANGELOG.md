# Changelog

## [2.0.0] - 2026-06-10

### Added

- **80-subject** training set (8,552 axial slices)
- Combined **80% SSIM + 20% MSE** loss via `pytorch_msssim`
- Extended training: **75 epochs max**, early stopping **patience 10** (converged at epoch 53)

### Changed

- Test results: **34.88 dB PSNR** · **0.9737 SSIM** (+6.22 dB / +0.0849 vs bicubic)
- Updated experiment figures in `docs/assets/`
- `src/smore/losses.py` aligned with notebook `combined_loss()`
- README rewritten for v2.0 results

## [1.1.0] - 2026-06-02

### Added

- Subject-level split utility, CI smoke tests, `CITATION.cff`

## [1.0.0] - 2026-06-02

### Added

- Initial SMORE DIP Jupyter pipeline, documentation assets, and README
