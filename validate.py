import argparse
import yaml
import os
import glob
import csv
import re

from src.eval.seld_validator import SELDValidator


CONFIG_PATH = "/app/configs/mid_fusion.yaml"
CHECKPOINT_DIR = "/app/results/checkpoints/mid_fusion/"
MONITOR_PATH = "/app/results/val"


def load_args(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return argparse.Namespace(
        # Daten
        train_wav_txt=cfg["data"]["train_wav_txt"],
        feature_config=cfg["data"]["feature_config"],
        batch_size=cfg["data"]["batch_size"],
        train_wav_length=cfg["data"]["train_wav_length"],
        video_fps=cfg["data"]["video_fps"],
        # Features
        fft_size=cfg["features"]["fft_size"],
        stft_hop_size=cfg["features"]["stft_hop_size"],
        feature=cfg["features"]["feature"],
        # Modell
        class_num=cfg["model"]["class_num"],
        net=cfg["model"].get("net", "crnn"),
        # Validierung
        val=cfg["validation"]["val"],
        eval=cfg["validation"]["eval"],
        val_wav_txt=cfg["validation"]["val_wav_txt"],
        eval_wav_txt=cfg["validation"]["eval_wav_txt"],
        eval_model=cfg["validation"]["eval_model"],
        threshold_config=cfg["validation"]["threshold_config"],
        eval_wav_hop_length=cfg["validation"]["eval_wav_hop_length"],
        sampling_frequency=cfg["validation"]["sampling_frequency"],
    )


def get_iteration_from_filename(filename):
    match = re.search(r"params_(\d+)\.pth", filename)
    return int(match.group(1)) if match else 0


def main():
    args = load_args(CONFIG_PATH)
    
    # 1. Alle Checkpoints finden und nach Iterationszahl sortieren
    checkpoints = sorted(
        glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth")),
        key=get_iteration_from_filename
    )
    
    if not checkpoints:
        print(f"Keine Checkpoints in {CHECKPOINT_DIR} gefunden!")
        return

    print(f"Insgesamt {len(checkpoints)} Checkpoints gefunden. Starte Batch-Evaluierung...")
    print("=" * 80)
    
    # Validator nur einmal laden (spart massiv Zeit)
    validator = SELDValidator(args, monitor_path=MONITOR_PATH)
    
    results = []

    # 2. Schleife über alle 20 Checkpoints
    for ckpt_path in checkpoints:
        iteration = get_iteration_from_filename(ckpt_path)
        print(f">>> Evaluiere Iteration {iteration} ({os.path.basename(ckpt_path)}) ...")
        
        metrics, val_loss = validator.validation(ckpt_path)
        er, f, le, lr, seld_err = metrics
        results.append((iteration, er, f, le, lr, seld_err, val_loss))

    # 3. CSV-Datei speichern
    os.makedirs(MONITOR_PATH, exist_ok=True)
    csv_path = os.path.join(MONITOR_PATH, "mid_fusion_validation_curve.csv")
    
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "ER20", "F20", "LE", "LR", "SELD_Score", "Val_Loss"])
        for row in results:
            writer.writerow(row)

    # 4. Abschließende Übersicht im Terminal anzeigen
    print("\n" + "=" * 80)
    print("FINALE ERGEBNISSE - VALIDIERUNGSKURVE")
    print("=" * 80)
    print(f"{'Iter':<8} | {'ER20 (↓)':<8} | {'F20 (↑)':<8} | {'LE (↓)':<8} | {'LR (↑)':<8} | {'SELD (↓)':<8}")
    print("-" * 80)
    for row in results:
        it, er, f, le, lr, seld_err, _ = row
        print(f"{it:<8} | {er:<8.3f} | {f:<8.3f} | {le:<8.2f} | {lr:<8.3f} | {seld_err:<8.3f}")
    print("=" * 80)
    print(f"CSV erfolgreich gespeichert unter: {csv_path}")


if __name__ == "__main__":
    main()