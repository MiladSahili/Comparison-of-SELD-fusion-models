import os
import pandas as pd

CSV_PATH = "/app/results/val/mid_fusion_validation_curve.csv"


def main():
    if not os.path.exists(CSV_PATH):
        print(f"Datei nicht gefunden: {CSV_PATH}")
        return

    # 1. CSV-Datei einlesen
    df = pd.read_csv(CSV_PATH)

    # 2. Den besten Checkpoint nach SELD-Score (niedriger ist besser) finden
    best_idx = df["SELD_Score"].idxmin()
    best_row = df.loc[best_idx]

    iteration = int(best_row["Iteration"])
    er = best_row["ER20"]
    f20 = best_row["F20"] * 100 if best_row["F20"] <= 1.0 else best_row["F20"]
    le = best_row["LE"]
    lr = best_row["LR"] * 100 if best_row["LR"] <= 1.0 else best_row["LR"]
    seld_score = best_row["SELD_Score"]

    # 3. Übersicht im Terminal ausgeben
    print("=" * 65)
    print(f"BESTER CHECKPOINT GEFUNDEN: Iteration {iteration}")
    print("=" * 65)
    print(f"  ER20 (Error Rate)          : {er:.3f}")
    print(f"  F20  (F-Score)             : {f20:.1f} %")
    print(f"  LE   (Localization Error)  : {le:.1f} Grad")
    print(f"  LR   (Localization Recall) : {lr:.1f} %")
    print(f"  SELD-Score (Gesamt)        : {seld_score:.3f}")
    print("-" * 65)

    # 4. Fertiger LaTeX-Code zum Kopieren
    print("FERTIGE ZEILE FÜR DEINE LATEX-TABELLE:")
    latex_line = (
        f"    Mid-Fusion (Iter. {iteration}) & "
        f"{er:.3f} & "
        f"{f20:.1f}\\,\\% & "
        f"{le:.1f}^\\circ & "
        f"{lr:.1f}\\,\\% & "
        f"{seld_score:.3f} \\\\"
    )
    print(latex_line)
    print("=" * 65)


if __name__ == "__main__":
    main()