import os
import pandas as pd
import matplotlib.pyplot as plt

# Pfade zu deinen CSV-Dateien
VAL_CSV = "/app/results/val/mid_fusion_validation_curve.csv"
TRAIN_LOSS_CSV = "/app/results/val/tensorboard_Loss_train.csv"
OUTPUT_DIR = "/app/results/val/"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Daten einlesen
    df_val = pd.read_csv(VAL_CSV)
    df_train = pd.read_csv(TRAIN_LOSS_CSV)

    # Sortieren nach Iteration (zur Sicherheit)
    df_val = df_val.sort_values("Iteration")
    df_train = df_train.sort_values("Iteration")

    # Akademischen Plot-Stil aktivieren
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "figure.titlesize": 14
    })

    # =========================================================================
    # DIAGRAMM 1: Trainings- und Validierungs-Loss (Konvergenz)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    # Hochauflösende Trainingskurve aus TensorBoard
    ax.plot(
        df_train["Iteration"], df_train["Value"],
        label="Trainings-Loss (ADPIT)",
        color="tab:blue", alpha=0.6, linewidth=1.5
    )

    # Checkpoint-Punkte der Validierung
    ax.plot(
        df_val["Iteration"], df_val["Val_Loss"],
        label="Validierungs-Loss (Checkpoints)",
        color="tab:red", marker="o", linewidth=2, markersize=5
    )

    ax.set_title("Lernkurve: Trainings- und Validierungs-Loss", fontweight="bold")
    ax.set_xlabel("Trainings-Iterationen")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")  # Logarithmische Skala zeigt kleine Loss-Änderungen besser
    ax.legend(loc="upper right", frameon=True)

    loss_pdf = os.path.join(OUTPUT_DIR, "thesis_plot_loss.pdf")
    loss_png = os.path.join(OUTPUT_DIR, "thesis_plot_loss.png")
    plt.tight_layout()
    plt.savefig(loss_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(loss_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Diagramm 1 gespeichert: {loss_pdf}")

    # =========================================================================
    # DIAGRAMM 2: SELD-Metriken vs. Offizielle Baseline (STARSS23)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Werte aus DCASE-Skripten sind oft von 0 bis 1 -> in Prozent umrechnen falls nötig
    f20_vals = df_val["F20"] * 100 if df_val["F20"].max() <= 1.0 else df_val["F20"]
    lr_vals = df_val["LR"] * 100 if df_val["LR"].max() <= 1.0 else df_val["LR"]

    # --- Links: F-Score & Recall (Höher ist besser ↑) ---
    ax1.plot(df_val["Iteration"], f20_vals, marker="o", color="tab:blue", linewidth=2, label="Mid-Fusion F20")
    ax1.plot(df_val["Iteration"], lr_vals, marker="s", color="tab:cyan", linewidth=2, linestyle="--", label="Mid-Fusion LR")

    # Baseline-Referenzlinien (STARSS23 FOA Baseline)
    ax1.axhline(y=11.1, color="tab:blue", linestyle=":", alpha=0.8, label="Baseline F20 (11.1%)")
    ax1.axhline(y=35.2, color="tab:cyan", linestyle=":", alpha=0.8, label="Baseline LR (35.2%)")

    ax1.set_title("Erkennungs- & Lokalisierungsrate (↑ besser)", fontweight="bold")
    ax1.set_xlabel("Trainings-Iterationen")
    ax1.set_ylabel("Prozent (%)")
    ax1.legend(loc="lower right", frameon=True)

    # --- Rechts: Localization Error (LE) in Grad (Niedriger ist besser ↓) ---
    ax2.plot(df_val["Iteration"], df_val["LE"], marker="o", color="tab:red", linewidth=2, label="Mid-Fusion LE")

    # Baseline-Referenzlinie
    ax2.axhline(y=47.2, color="tab:red", linestyle=":", alpha=0.8, label="Baseline LE (47.2°)")

    ax2.set_title("Winkelfehler / Localization Error (↓ besser)", fontweight="bold")
    ax2.set_xlabel("Trainings-Iterationen")
    ax2.set_ylabel("Grad (°)")
    ax2.legend(loc="upper right", frameon=True)

    metrics_pdf = os.path.join(OUTPUT_DIR, "thesis_plot_metrics.pdf")
    metrics_png = os.path.join(OUTPUT_DIR, "thesis_plot_metrics.png")
    plt.tight_layout()
    plt.savefig(metrics_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(metrics_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Diagramm 2 gespeichert: {metrics_pdf}")

if __name__ == "__main__":
    main()