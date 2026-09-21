# Smoke-Test for mid_transformer model architecture
import yaml
import argparse
import torch
from src.training.trainer import _create_net


def load_args(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return argparse.Namespace(
        feature_config=cfg["data"]["feature_config"],
        feature=cfg["features"]["feature"],
        class_num=cfg["model"]["class_num"],
        net=cfg["experiment"].get("fusion", "crnn"),
    )


def main():
    args = load_args("configs/mid_transformer.yaml")
    print(f"net={args.net}, feature={args.feature}, class_num={args.class_num}")

    net = _create_net(args)
    net.eval()

    # Synthetic inputs matching a 1.27 s clip:
    # audio spec (B, C, F, T) = (2, 7, 257, 128)
    # visual feature (B, 2, 6, 37) = (2, 2, 6, 37)
    x_a = torch.randn(2, 7, 257, 128)
    x_v = torch.randn(2, 2, 6, 37)

    with torch.no_grad():
        out = net(x_a, x_v)

    print("Output shape:", out.shape)
    expected = (2, 3, 3, 13, 128)
    assert out.shape == expected, f"Expected {expected}, got {out.shape}"
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
