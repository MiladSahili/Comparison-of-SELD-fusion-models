import argparse
import torch
from torch.utils.data import DataLoader
from utils.dataloader import SELDDataSet, create_data_loader


def main():
    print("=== START SANITY CHECK FOR DATALOADER ===")

    # Args passend zu SELDDataSet.__init__(self, args, object_detection)
    args = argparse.Namespace(
        train_wav_txt="/app/data/dev_train.txt",
        feature_config="/app/configs/feature.json",
        batch_size=4,
        train_wav_length=1.27,   # Sekunden pro Sample
        fft_size=512,
        stft_hop_size=240,
        feature="amp_phasediff",
        class_num=13,
    )

    # 1. Dataset initialisieren
    # object_detection=None weil wir erst nur Audio/Label testen
    print("Initialisiere Dataset...")
    try:
        dataset = SELDDataSet(args, object_detection=None)
        print(f"✅ Dataset initialisiert. __len__ = {len(dataset)}")
    except Exception as e:
        print(f"❌ Fehler bei Dataset-Initialisierung: {e}")
        return

    # 2. DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 0 wegen cv2 + RAM-dict nicht thread-safe
    )

    # 3. Ein Batch laden und Shapes prüfen
    print("Lade erstes Batch...")
    try:
        for input_spec, frame_out, label, name in dataloader:
            print("\n🎉 ERFOLG! Erstes Batch geladen.")
            print(f"  input_spec shape : {input_spec.shape}")
            # Erwartet: (batch, channels, freq_bins, frames)
            # bei amp_phasediff: channels = wav_ch + (wav_ch-1) = 7 bei FOA (4+3)

            print(f"  frame_out shape  : {frame_out.shape}")
            # Erwartet: (batch, ...) — Output von object_detection.box2dist()

            print(f"  label shape      : {label.shape}")
            # Erwartet: (batch, frames, classes, 3) — Multi-ACCDOA

            print(f"  name[0]          : {name[0]}")
            break

    except Exception as e:
        print(f"\n❌ Fehler beim Laden des Batches: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== ENDE SANITY CHECK ===")


if __name__ == "__main__":
    main()