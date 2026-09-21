import torch
import torch.nn as nn
import json

from src.models.midlevel.net_util import interpolate
from src.models.midlevel.net_seld import CNN3


def create_net_seld(args):
    with open(args.feature_config, 'r') as f:
        feature_config = json.load(f)
    if args.net == 'audio_only_matched':
        Net = AudioOnlyCRNN_Matched(class_num=args.class_num,
                                    in_channels=feature_config[args.feature]["ch"])
    return Net


class AudioOnlyCRNN_Matched(nn.Module):
    """Capacity-matched Audio-Only-Baseline.

    Identischer Aufbau wie AudioOnlyCRNN (CNN3 -> bidir. GRU -> FC ->
    Multi-ACCDOA), aber mit GRU hidden_size=266 statt 256. Dadurch hat das
    Modell ~666k Parameter und damit praktisch dieselbe Kapazitaet wie das
    Late-Fusion-Modell (~669k = Audio-Branch ~629k + Visual-Branch ~40k).

    Zweck: Kontrollexperiment – falls Late Fusion besser ist als diese
    Baseline, ist der Gewinn nicht allein durch mehr Parameter erklaerbar.
    """
    def __init__(self, class_num, in_channels, interp_ratio=16):
        super().__init__()
        self.class_num = class_num
        self.interp_ratio = interp_ratio

        self.audio_encoder = CNN3(in_channels=in_channels, out_channels=64)
        self.gru = nn.GRU(input_size=64, hidden_size=266,
                          num_layers=1, batch_first=True, bidirectional=True)
        self.fc_xyz = nn.Linear(2 * 266, 3 * 3 * self.class_num, bias=True)

    def forward(self, x_a, x_v=None):          # x_v wird ignoriert
        x_a = x_a.transpose(2, 3)
        b, c, t, f = x_a.size()
        x_a = self.audio_encoder(x_a)
        x_a = torch.mean(x_a, dim=3)           # (B, 64, T)

        x = x_a.transpose(1, 2)                # (B, T, 64)
        self.gru.flatten_parameters()
        (x, _) = self.gru(x)
        x = self.fc_xyz(x)
        x = interpolate(x, self.interp_ratio)
        x = x.transpose(1, 2)
        return x.view(-1, 3, 3, self.class_num, t)
