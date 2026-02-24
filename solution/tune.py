"""
Hyperparameter tuning: try a grid of (hidden_size, lr, batch_size, num_epochs),
train each combo, evaluate on valid.parquet, save the best model by validation MSE.
Saves progress after each run so you can resume if you stop and run tune.py again.
Run from solution/: python tune.py
"""
import json
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

import torch
import preprocess
import train

# --- Grid: values to try. Add more or change to expand the search. ---
HIDDEN_SIZES = [64, 128]       # LSTM hidden units; more = more capacity, slower.
LEARNING_RATES = [5e-4, 1e-3]  # Lower often more stable if loss was increasing.
BATCH_SIZES = [256]
NUM_EPOCHS_TUNE = 5            # Epochs per run (low = faster tuning).

TUNE_CHECKPOINT_PATH = os.path.join(CURRENT_DIR, "tune_checkpoint.pth")


def _all_combos():
    """List of (hidden_size, lr, batch_size) in same order as the loop."""
    out = []
    for hidden_size in HIDDEN_SIZES:
        for lr in LEARNING_RATES:
            for batch_size in BATCH_SIZES:
                out.append((hidden_size, lr, batch_size))
    return out


def _save_tune_checkpoint(completed_run_ids, best_valid_mse, best_params, best_model):
    """Save so we can resume later."""
    state = best_model.state_dict() if best_model is not None else None
    torch.save(
        {
            "completed_run_ids": completed_run_ids,
            "best_valid_mse": best_valid_mse,
            "best_params": best_params,
            "best_model_state": state,
        },
        TUNE_CHECKPOINT_PATH,
    )


def main():
    # Check data exists.
    if not os.path.exists(train.TRAIN_PATH):
        print(f"Train file not found: {train.TRAIN_PATH}")
        return
    if not os.path.exists(train.VALID_PATH):
        print(f"Valid file not found: {train.VALID_PATH} (need it for tuning)")
        return

    # Fit preprocessing once; same mean/std used for all runs and for valid.
    print("Fitting preprocessing on training data...")
    mean, std = preprocess.fit_normalization_from_parquet(train.TRAIN_PATH)
    scale_path = os.path.join(CURRENT_DIR, "lstm_scale.npz")
    preprocess.save_scale_params(scale_path, mean, std, preprocess.ROLLING_WINDOW)

    combos = _all_combos()
    completed_run_ids = []
    best_valid_mse = float("inf")
    best_params = None
    best_model = None

    # If starting fresh (no tune checkpoint), clear any leftover per-run checkpoints.
    if not os.path.exists(TUNE_CHECKPOINT_PATH):
        for f in os.listdir(CURRENT_DIR):
            if f.startswith("tune_run_") and f.endswith(".pth"):
                try:
                    os.remove(os.path.join(CURRENT_DIR, f))
                except Exception:
                    pass

    # Resume from checkpoint if present.
    if os.path.exists(TUNE_CHECKPOINT_PATH):
        print(f"Resuming from {TUNE_CHECKPOINT_PATH}")
        ckpt = torch.load(TUNE_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        completed_run_ids = ckpt.get("completed_run_ids", [])
        best_valid_mse = ckpt.get("best_valid_mse", float("inf"))
        best_params = ckpt.get("best_params")
        state = ckpt.get("best_model_state")
        if best_params is not None and state is not None:
            best_model = train.LSTMPredictor(hidden_size=best_params["hidden_size"])
            best_model.load_state_dict(state)
        print(f"  Completed runs: {len(completed_run_ids)}/{len(combos)}. Best valid MSE so far: {best_valid_mse:.6f}")

    # Loop over all combinations; skip already completed.
    for run_id, (hidden_size, lr, batch_size) in enumerate(combos):
        if run_id in completed_run_ids:
            print(f"\n--- Run {run_id + 1}: hidden={hidden_size}, lr={lr}, batch={batch_size} (skipped, already done) ---")
            continue
        print(f"\n--- Run {run_id + 1}: hidden={hidden_size}, lr={lr}, batch={batch_size}, epochs={NUM_EPOCHS_TUNE} ---")
        run_ckpt = os.path.join(CURRENT_DIR, f"tune_run_{run_id}.pth")
        model = train.run_training(
            mean, std,
            hidden_size=hidden_size,
            lr=lr,
            batch_size=batch_size,
            num_epochs=NUM_EPOCHS_TUNE,
            run_checkpoint_path=run_ckpt,
        )
        if os.path.exists(run_ckpt):
            try:
                os.remove(run_ckpt)
            except Exception:
                pass
        valid_mse = train.evaluate_valid_mse(
            model, train.VALID_PATH, mean, std, preprocess.ROLLING_WINDOW
        )
        print(f"  valid MSE: {valid_mse:.6f}")
        if valid_mse < best_valid_mse:
            best_valid_mse = valid_mse
            best_params = {
                "hidden_size": hidden_size,
                "lr": lr,
                "batch_size": batch_size,
                "num_epochs": NUM_EPOCHS_TUNE,
            }
            best_model = model
            print(f"  -> new best")
        completed_run_ids.append(run_id)
        _save_tune_checkpoint(completed_run_ids, best_valid_mse, best_params, best_model)
        print(f"  checkpoint saved ({len(completed_run_ids)}/{len(combos)} runs done)")

    if best_model is None:
        print("No run completed.")
        return

    # Save best model and its hyperparams (solution.py must use same hidden_size when loading).
    print(f"\nBest valid MSE: {best_valid_mse:.6f}")
    print("Best params:", best_params)
    out_path = os.path.join(CURRENT_DIR, "lstm_weights.pth")
    torch.save(best_model.state_dict(), out_path)
    params_path = os.path.join(CURRENT_DIR, "lstm_best_params.json")
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Best model saved to {out_path}")
    print(f"Best params saved to {params_path} (use same hidden_size in solution.py when loading)")


if __name__ == "__main__":
    main()
