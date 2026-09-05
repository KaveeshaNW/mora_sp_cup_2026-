"""
finetune.py
Fine-tuning routine with Cosine Annealing, Composite Loss, and ONNX Export
"""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

from dataset import LowLightDenoiseDataset
from train_utils import CompositeLoss, inject_synthetic_impulse_noise


def export_to_onnx(model: torch.nn.Module, export_path: Path):
    """Exports model with dynamic spatial axes for flexible batch tiling."""
    model.eval()
    model.cpu()
    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(export_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        },
    )
    print(f"[+] Successfully exported model to {export_path}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # 1. Dataset & Loaders
    full_dataset = LowLightDenoiseDataset(
        gt_dir=args.gt_dir, noisy_dir=args.noisy_dir, patch_size=256, is_train=True
    )
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # 2. Model, Loss, Optimizer
    # Replace NAFNet with your imported network definition
    from your_model_file import NAFNet  # Adjust import to your codebase
    model = NAFNet().to(device)

    if args.pretrained_weights:
        model.load_state_dict(torch.load(args.pretrained_weights, map_location=device))
        print(f"[+] Loaded weights from {args.pretrained_weights}")

    criterion = CompositeLoss(alpha_l1=0.7, beta_ssim=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    output_onnx_path = Path("scripts/nafnet_denoise.onnx")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for noisy_batch, gt_batch in train_loader:
            noisy_batch = noisy_batch.to(device)
            gt_batch = gt_batch.to(device)

            # Synthetic impulse injection (0.8% probability)
            noisy_corrupted = inject_synthetic_impulse_noise(noisy_batch, prob=0.008)

            optimizer.zero_grad()
            preds = model(noisy_corrupted)
            loss = criterion(preds, gt_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * noisy_batch.size(0)

        scheduler.step()
        train_loss /= len(train_set)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_noisy, val_gt in val_loader:
                val_noisy = val_noisy.to(device)
                val_gt = val_gt.to(device)
                val_preds = model(val_noisy)
                val_loss += criterion(val_preds, val_gt).item() * val_noisy.size(0)

        val_loss /= len(val_set)
        print(
            f"Epoch [{epoch}/{args.epochs}] - Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}"
        )

        # Save checkpoint on lowest validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"[*] Best checkpoint updated at epoch {epoch}")

    # 4. Final Export
    model.load_state_dict(torch.load("best_model.pth"))
    export_to_onnx(model, output_onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_dir", type=Path, default=Path("competition_data/public/ground_truth")
    )
    parser.add_argument(
        "--noisy_dir", type=Path, default=Path("competition_data/public/noisy")
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pretrained_weights", type=str, default=None)
    args = parser.parse_args()

    train(args)
