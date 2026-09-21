import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.midlevel.net_util import interpolate


def create_net_seld(args):
    with open(args.feature_config, "r") as f:
        feature_config = json.load(f)
    in_channels = feature_config[args.feature]["ch"]

    if args.net in ("visual_only_2", "late_2"):
        Net = VisualOnlyCRNN_2(class_num=args.class_num)
    else:
        raise ValueError(f"Unknown net: {args.net}")
    return Net


class VisualOnlyCRNN_2(nn.Module):
    """Visual-only SELD model for decision-level late fusion.

    Same output format (Multi-ACCDOA) as the audio and audio-visual models, so
    that scores can be averaged at inference time. The GRU has been removed 
    as the visual input is static per chunk.
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

        # Output Head: Nimmt jetzt direkt die 64 Dimensionen vom Vision-Encoder
        # (512 war vorher der Output der bidirektionalen GRU)
        self.fc_xyz = nn.Linear(vis_embed_size, 3 * 3 * class_num, bias=True)

    def forward(self, x_a, x_v):
        # x_a: (B, 7, 257, 128) -- nur fuer die Clip-Laenge genutzt
        t = x_a.size(-1)                        # 128 time frames

        x = x_v.view(x_v.size(0), -1)           # (B, 444)
        x = self.vision_encoder(x)              # (B, 64)
        
        # Direkt in die Klassifizierungs-Schicht (ohne GRU)
        x = self.fc_xyz(x)                      # (B, 117)
        
        # Jetzt kopieren wir das finale statische Ergebnis für die 8 Zeitschritte
        x = torch.unsqueeze(x, dim=-1)          # (B, 117, 1)
        x = x.expand(-1, -1, 8)                 # (B, 117, 8) -- künstliche Zeitachse erzeugen

        # Interpolation auf die finale Audio-Auflösung
        x = interpolate(x, self.interp_ratio)   # (B, 117, 128)
        
        return x.view(-1, 3, 3, self.class_num, t)