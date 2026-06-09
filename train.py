import argparse
import yaml
from src.data.dataloader import create_data_loader



def load_args(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # die verschachtelte YAML in ein flaches args-Objekt ueberfuehren,
    # so wie der Dataloader es erwartet
    return argparse.Namespace(
        train_wav_txt=cfg["data"]["train_wav_txt"],
        feature_config=cfg["data"]["feature_config"],
        batch_size=cfg["data"]["batch_size"],
        train_wav_length=cfg["data"]["train_wav_length"],
        video_fps=cfg["data"]["video_fps"],
        fft_size=cfg["features"]["fft_size"],
        stft_hop_size=cfg["features"]["stft_hop_size"],
        feature=cfg["features"]["feature"],
        class_num=cfg["model"]["class_num"],
    )


def main():
    
    args = load_args("/app/configs/mid_fusion.yaml")

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