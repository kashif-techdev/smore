#!/usr/bin/env python3
"""Prepare v2 notebook for git: strip outputs, extract figures, clean cells."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "SMORE_DIP_Project (1).ipynb"
DST = ROOT / "notebooks" / "SMORE_DIP_Project.ipynb"
ASSETS = ROOT / "docs" / "assets"

FIGURE_NAMES = [
    "real_mri_slices",
    "degradation_simulation",
    "training_curves",
    "test_comparison",
    "ablation_study",
    "edge_sharpness",
]

TRAINING_CELL_START = """# ── Hyperparameters ────────────────────────────────────────
NUM_EPOCHS      = 75       # more epochs for better convergence
LEARNING_RATE   = 1e-3
PATIENCE        = 10       # give the model more time to improve
SSIM_ALPHA      = 0.8      # 80% SSIM-driven, 20% MSE

# ── Combined SSIM + MSE Loss (replaces pure nn.MSELoss) ───
def combined_loss(pred, target, alpha=SSIM_ALPHA):
    mse_val = F.mse_loss(pred, target)
    ssim_val = 1.0 - ssim_metric(pred, target, data_range=1.0, size_average=True)
    return alpha * ssim_val + (1.0 - alpha) * mse_val

criterion = combined_loss

# ── Optimizer & Scheduler ─────────────────────────────────
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# ── Training history ──────────────────────────────────────
history = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": []}

best_val_loss = float("inf")
patience_count = 0
best_model_wts = None


def evaluate(model, loader):
    \"\"\"Run model on loader, return avg loss, PSNR, SSIM.\"\"\"
    model.eval()
    total_loss, total_psnr, total_ssim, n = 0, 0, 0, 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            for i in range(inputs.size(0)):
                pred = outputs[i, 0].cpu().numpy()
                tgt = targets[i, 0].cpu().numpy()
                total_psnr += peak_signal_noise_ratio(tgt, pred, data_range=1.0)
                total_ssim += structural_similarity(tgt, pred, data_range=1.0)
                n += 1

    return total_loss / n, total_psnr / n, total_ssim / n


# ── Training Loop ─────────────────────────────────────────
print(f"Starting training for {NUM_EPOCHS} epochs  |  SSIM alpha={SSIM_ALPHA}  MSE alpha={1-SSIM_ALPHA}")
print(f"   Batch size    : {BATCH_SIZE}")
print(f"   Learning rate : {LEARNING_RATE}")
print(f"   Device        : {device}")
print("-" * 65)
print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>9} | {'Val PSNR':>9} | {'Val SSIM':>9}")
print("-" * 65)

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    train_loss = running_loss / len(train_dataset)

    val_loss, val_psnr, val_ssim = evaluate(model, val_loader)
    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_psnr"].append(val_psnr)
    history["val_ssim"].append(val_ssim)

    print(f"{epoch:>6} | {train_loss:>10.6f} | {val_loss:>9.6f} | {val_psnr:>9.4f} | {val_ssim:>9.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_count = 0
        best_model_wts = {k: v.clone() for k, v in model.state_dict().items()}
        torch.save(best_model_wts, "best_smore_model.pth")
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"\\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

print("-" * 65)
print("Training complete!")
print(f"   Best val loss : {best_val_loss:.6f}")
print(f"   Best val PSNR : {max(history['val_psnr']):.4f} dB")
print(f"   Best val SSIM : {max(history['val_ssim']):.4f}")

model.load_state_dict(best_model_wts)
print("Best model weights loaded")
"""


def extract_figures(nb: dict) -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    idx = 0
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            if out.get("output_type") != "display_data":
                continue
            data = out.get("data", {})
            if "image/png" not in data:
                continue
            name = FIGURE_NAMES[idx] if idx < len(FIGURE_NAMES) else f"figure_{idx}"
            (ASSETS / f"{name}.png").write_bytes(base64.b64decode(data["image/png"]))
            idx += 1
    return idx


def patch_cells(nb: dict) -> None:
    for cell in nb["cells"]:
        if not cell.get("source"):
            continue
        text = "".join(cell["source"])

        if "%pip install nibabel scikit-image tqdm kagglehub" in text:
            if "pytorch-msssim" not in text:
                text = text.replace(
                    "%pip install nibabel scikit-image tqdm kagglehub -q",
                    "%pip install nibabel scikit-image tqdm kagglehub pytorch-msssim -q",
                )

        if "from torch.utils.data import Dataset, DataLoader" in text and "pytorch_msssim" not in text:
            if "ssim_metric" not in text and "NUM_EPOCHS" not in text:
                text = text.replace(
                    "from torch.utils.data import Dataset, DataLoader, random_split\n",
                    "from torch.utils.data import Dataset, DataLoader, random_split\n"
                    "from pytorch_msssim import ssim as ssim_metric\n",
                )

        if "def combined_loss(pred, target" in text:
            cell["source"] = [line + "\n" for line in TRAINING_CELL_START.split("\n")[:-1]] + [
                TRAINING_CELL_START.split("\n")[-1] + "\n"
            ]
            continue

        if "SMORE DIP PROJECT — FINAL RESULTS SUMMARY" in text:
            text = text.replace(
                "   Loss         : MSE\n",
                "   Loss         : Combined SSIM+MSE (alpha={SSIM_ALPHA})\n",
            )

        if cell.get("cell_type") == "markdown" and "optional SSIM regularization" in text:
            text = text.replace(
                "- **Loss**: MSE (pixel-wise reconstruction) + optional SSIM regularization",
                "- **Loss**: Combined **80% SSIM + 20% MSE** (`pytorch_msssim`)",
            )

        if text != "".join(cell.get("source", [])):
            lines = text.splitlines(keepends=True)
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            cell["source"] = lines


def main() -> None:
    raw_nb = json.loads(SRC.read_text(encoding="utf-8"))
    n_figs = extract_figures(raw_nb)
    print(f"Extracted {n_figs} figures to {ASSETS}")

    nb = json.loads(SRC.read_text(encoding="utf-8"))
    patch_cells(nb)

    for cell in nb["cells"]:
        cell["outputs"] = []
        cell["execution_count"] = None
    nb["metadata"].pop("widgets", None)

    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote cleaned notebook ({DST.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
