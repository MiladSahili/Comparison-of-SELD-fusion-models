import argparse, yaml, torch
from torchinfo import summary

from src.data.dataloader import create_data_loader
from src.models.midlevel.net_seld import create_net_seld


def load_args(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
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
        net=cfg["model"].get("net", "crnn"),
    )


def main():
    args = load_args("/app/configs/mid_fusion.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloader = create_data_loader(args)
    spec, vis, label, names = next(iter(dataloader))

    model = create_net_seld(args).to(device)

    summary(
        model,
        input_data=(spec.to(device), vis.to(device)),
        col_names=["input_size", "output_size", "num_params"],
        depth=4,
    )


if __name__ == "__main__":
    main()