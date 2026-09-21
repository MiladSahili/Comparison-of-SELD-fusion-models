import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.midlevel.net_util import interpolate


def create_net_seld(args):
    with open(args.feature_config, "r") as f:
        feature_config = json.load(f)
    in_channels = feature_config[args.feature]["ch"]

    if args.net in ("crnn", "visual_only", "late"):
        Net = VisualOnlyCRNN(class_num=args.class_num)
    else:
        raise ValueError(f"Unknown net: {args.net}")
    return Net


class VisualOnlyCRNN(nn.Module):
    """Visual-only SELD model for decision-level late fusion.

    Same output format (Multi-ACCDOA) and same downstream structure
    (GRU + FC + interpolate) as the audio and audio-visual models, so
    that scores can be averaged at inference time (Snoek et al., 2005;
    Baltrusaitis et al., 2019). The audio input x_a is only used to
    determine the clip length (t = 128 frames).
    """

    def __init__(self, class_num, interp_ratio=16):
        super().__init__()
        self.class_num = class_num
        self.interp_ratio = interp_ratio

        # Visual encoder (identisch zum visuellen Zweig der Mid-Fusion-Baseline)
        vis_in_size = 2 * 6 * 37                # 444 Gaussian vector features
        vis_embed_size = 64
        self.vision_encoder = nn.Sequential(
            nn.Linear(vis_in_size, vis_embed_size),
            nn.Linear(vis_embed_size, vis_embed_size),
        )

        # Temporal modelling + output head (identisch zur Baseline)
        self.gru = nn.GRU(input_size=vis_embed_size, hidden_size=256,
                          num_layers=1, batch_first=True, bidirectional=True)
        self.fc_xyz = nn.Linear(512, 3 * 3 * class_num, bias=True)

    def forward(self, x_a, x_v):
        # x_a: (B, 7, 257, 128) -- nur fuer die Clip-Laenge genutzt
        t = x_a.size(-1)                        # 128 time frames

        x = x_v.view(x_v.size(0), -1)           # (B, 444)
        x = self.vision_encoder(x)              # (B, 64)
        x = torch.unsqueeze(x, dim=1)           # (B, 1, 64)
        x = x.expand(-1, 8, -1)                 # (B, 8, 64) -- 8 Schritte wie Baseline

        self.gru.flatten_parameters()
        x, _ = self.gru(x)                      # (B, 8, 512)
        x = self.fc_xyz(x)                      # (B, 8, 117)
        x = x.transpose(1, 2)                   # (B, 117, 8)

        x = interpolate(x, self.interp_ratio)   # (B, 117, 128)
        return x.view(-1, 3, 3, self.class_num, t)