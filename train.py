import argparse

from utils.dataloader import create_data_loader


def main():
    args = argparse.Namespace(
        train_wav_txt="/app/data/dev_train_clean.txt",
        feature_config="/app/configs/feature.json",
        batch_size=4,
        train_wav_length=1.27,
        fft_size=512,
        stft_hop_size=240,
        feature="amp_phasediff",
        class_num=13,
    )

    dataloader = create_data_loader(args)

    spec, vis, label, names = next(iter(dataloader))

    print("=== Batch-Shapes ===")
    print("input_spec      :", spec.shape)    # (B, C, T, F)
    print("frame_out_float :", vis.shape)     # (B, 2, 6, 37)
    print("label_float     :", label.shape)
    print("names           :", names)
    print("====================")


if __name__ == "__main__":
    main()