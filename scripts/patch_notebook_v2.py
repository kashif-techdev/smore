#!/usr/bin/env python3
"""Apply v2 notebook patches (subject splits + SSIM loss)."""

import json
from pathlib import Path


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "notebooks" / "SMORE_DIP_Project.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))

    for cell in nb["cells"]:
        if not cell.get("source"):
            continue
        text = "".join(cell["source"])

        if (
            "from torch.utils.data import Dataset, DataLoader, random_split" in text
            and "sys.path.insert" not in text
        ):
            old = (
                '    print("   ⚠️ No GPU found. Training will be slow. '
                'Go to Runtime → Change runtime type → T4 GPU")'
            )
            if old in text:
                text = text.replace(
                    old,
                    old
                    + """

# Repository root (for src/smore imports in local clone or Colab)
import sys
from pathlib import Path
_repo_root = Path.cwd()
if not (_repo_root / "src").is_dir():
    _repo_root = _repo_root.parent
if (_repo_root / "src").is_dir() and str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))""",
                )

        if "all_slices = []" in text and "slices_by_subject" not in text:
            text = text.replace(
                """all_slices = []
selected_files = valid_nii_files[:NUM_SUBJECTS]

for fpath in tqdm(selected_files, desc="Loading volumes"):
    slices = extract_slices(fpath)
    all_slices.extend(slices)

print(f"\\n✅ Total 2D slices extracted : {len(all_slices)}")""",
                """slices_by_subject = []
selected_files = valid_nii_files[:NUM_SUBJECTS]

for fpath in tqdm(selected_files, desc="Loading volumes"):
    slices = extract_slices(fpath)
    if slices:
        slices_by_subject.append(slices)

all_slices = [sl for subject in slices_by_subject for sl in subject]

print(f"\\n✅ Total 2D slices extracted : {len(all_slices)}")
print(f"   Subjects with valid slices : {len(slices_by_subject)}")""",
            )

        if "shuffled_idx = rng.permutation(total)" in text:
            text = text.replace(
                """# ── Train / Validation / Test split ────────────────────────
total      = len(all_slices)
n_train    = int(0.70 * total)
n_val      = int(0.15 * total)
n_test     = total - n_train - n_val

# Deterministic shuffle
rng = np.random.default_rng(seed=42)
shuffled_idx = rng.permutation(total)

train_slices = [all_slices[i] for i in shuffled_idx[:n_train]]
val_slices   = [all_slices[i] for i in shuffled_idx[n_train:n_train+n_val]]
test_slices  = [all_slices[i] for i in shuffled_idx[n_train+n_val:]]

print(f"✅ Dataset split:")
print(f"   Train : {len(train_slices):,} slices")
print(f"   Val   : {len(val_slices):,} slices")
print(f"   Test  : {len(test_slices):,} slices")""",
                """# ── Subject-level train / val / test split (no patient leakage) ──
from smore.splits import subject_level_split

split = subject_level_split(slices_by_subject, train_ratio=0.70, val_ratio=0.15, seed=42)
train_slices = split["train_slices"]
val_slices   = split["val_slices"]
test_slices  = split["test_slices"]

print("✅ Subject-level dataset split (no overlap between partitions):")
print(f"   Train : {split['train_subjects']} subjects | {len(train_slices):,} slices")
print(f"   Val   : {split['val_subjects']} subjects | {len(val_slices):,} slices")
print(f"   Test  : {split['test_subjects']} subjects | {len(test_slices):,} slices")""",
            )

        if cell.get("cell_type") == "markdown" and "optional SSIM regularization" in text:
            text = text.replace(
                "- **Loss**: MSE (pixel-wise reconstruction) + optional SSIM regularization",
                "- **Loss**: Combined **MSE + SSIM** (`CombinedLoss`, weights 1.0 / 0.2)",
            )

        if "criterion  = nn.MSELoss()" in text and text.strip().startswith("import os"):
            marker = "# ── Hyperparameters ────────────────────────────────────────"
            idx = text.find(marker)
            if idx != -1:
                text = text[idx:]
            text = text.replace(
                """# ── Hyperparameters ────────────────────────────────────────
NUM_EPOCHS      = 20       # Increase to 50+ for better results
LEARNING_RATE   = 1e-3
PATIENCE        = 5        # Early stopping patience

# ── Loss, Optimizer, Scheduler ────────────────────────────
criterion  = nn.MSELoss()""",
                """# ── Hyperparameters ────────────────────────────────────────
NUM_EPOCHS      = 20       # Increase to 50+ for better results
LEARNING_RATE   = 1e-3
PATIENCE        = 5        # Early stopping patience
MSE_WEIGHT      = 1.0
SSIM_WEIGHT     = 0.2      # Structural similarity term (v2)

# ── Loss, Optimizer, Scheduler ────────────────────────────
from smore.losses import CombinedLoss

criterion = CombinedLoss(mse_weight=MSE_WEIGHT, ssim_weight=SSIM_WEIGHT)""",
            )

        if "SMORE DIP PROJECT — FINAL RESULTS SUMMARY" in text:
            text = text.replace(
                "   Subjects : {NUM_SUBJECTS}\n"
                "   Slices   : {len(all_slices):,} total  |  Train: {len(train_slices):,}  Val: {len(val_slices):,}  Test: {len(test_slices):,}\n",
                "   Subjects : {NUM_SUBJECTS} ({split['train_subjects']} train / {split['val_subjects']} val / {split['test_subjects']} test)\n"
                "   Slices   : {len(all_slices):,} total  |  Train: {len(train_slices):,}  Val: {len(val_slices):,}  Test: {len(test_slices):,}\n"
                "   Split    : Subject-level (no patient leakage)\n",
            )
            text = text.replace(
                "   Loss         : MSE\n",
                "   Loss         : MSE + SSIM (w={SSIM_WEIGHT})\n",
            )

        if "| Add Perceptual/SSIM loss alongside MSE |" in text:
            text = text.replace(
                "| Add Perceptual/SSIM loss alongside MSE | Better structural quality |",
                "| Tune `SSIM_WEIGHT` (default 0.2) | Balance PSNR vs structural quality |",
            )

        if text != "".join(cell["source"]):
            lines = text.splitlines(keepends=True)
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            cell["source"] = lines

    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Patched", path)


if __name__ == "__main__":
    main()
