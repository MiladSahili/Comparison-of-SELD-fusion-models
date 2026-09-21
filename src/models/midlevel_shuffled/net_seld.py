import torch
import torch.nn as nn
import json

from src.models.midlevel.net_util import interpolate
from src.models.midlevel.net_seld import CNN3


def create_net_seld(args):
    with open(args.feature_config, 'r') as f:
        feature_config = json.load(f)
    if args.net in ('mid_shuffled', 'crnn_shuffled'):
        Net = AudioVisualCRNN_Shuffled(class_num=args.class_num,
                                       in_channels=feature_config[args.feature]["ch"])
    return Net


class AudioVisualCRNN_Shuffled(nn.Module):
    """Mid-Fusion-Kontrollexperiment mit geshuffelten visuellen Features.

    Architektur identisch zur Mid-Fusion (AudioVisualCRNN): separate
    Audio-CNN- und Visual-MLP-Encoder, Konkatenation der 64-dim Embeddings,
    geteiltes bidirektionales GRU. Der einzige Unterschied: die visuellen
    Features werden vor jedem Forward ueber die Batch-Dimension permutiert,
    sodass Audio- und Visual-Eingang eines Samples nicht mehr zur selben
    Szene gehoeren.

    Zweck: Positiv-Kontrolle zu late_fusion_shuffled. Erwartung: Das
    Ergebnis faellt deutlich ab (vs. ungeschuffeltes Mid), da das shared
    GRU die korrekte Audio-Visual-Korrespondenz aktiv nutzt. Zusammen mit
    den Late-Kontrollen ergibt sich die 2x2-Matrix, die zeigt, dass der
    visuelle Nutzen an der Feature-Level-Fusionsstelle liegt.

    Das Shuffeln passiert IMMER (auch im Eval-Modus), damit die Validierung
    dieselbe Bedingung sieht wie das Training.
    """

    def __init__(self, class_num, in_channels, interp_ratio=16):
        super().__init__()
        self.class_num = class_num
        self.interp_ratio = interp_ratio

        # Audio
        aud_embed_size = 64
        self.audio_encoder = CNN3(in_channels=in_channels, out_channels=aud_embed_size)

        # Visual
        vis_embed_size = 64
        vis_in_size = 2 * 6 * 37
        project_vis_embed_fc1 = nn.Linear(vis_in_size, vis_embed_size)
        project_vis_embed_fc2 = nn.Linear(vis_embed_size, vis_embed_size)
        self.vision_encoder = nn.Sequential(project_vis_embed_fc1,
                                            project_vis_embed_fc2)

        # Audio-Visual
        in_size_gru = aud_embed_size + vis_embed_size
        self.gru = nn.GRU(input_size=in_size_gru, hidden_size=256,
                          num_layers=1, batch_first=True, bidirectional=True)
        self.fc_xyz = nn.Linear(512, 3 * 3 * self.class_num, bias=True)

    def forward(self, x_a, x_v):
        # Visuelle Features ueber die Batch-Dimension shuffeln:
        # Sample i bekommt das Visual-Feature von Sample perm[i].
        idx = torch.randperm(x_v.size(0), device=x_v.device)
        x_v = x_v[idx]

        x_a = x_a.transpose(2, 3)
        b_a, c_a, t_a, f_a = x_a.size()
        b, c, t, f = b_a, c_a, t_a, f_a
        x_a = self.audio_encoder(x_a)
        x_a = torch.mean(x_a, dim=3)  # (B, 64, T)

        x_v = x_v.view(x_v.size(0), -1)
        x_v = self.vision_encoder(x_v)
        x_v = torch.unsqueeze(x_v, dim=-1).repeat(1, 1, 8)

        x = torch.cat((x_a, x_v), 1)

        x = x.transpose(1, 2)
        self.gru.flatten_parameters()
        (x, _) = self.gru(x)

        x = self.fc_xyz(x)
        x = interpolate(x, self.interp_ratio)
        x = x.transpose(1, 2)
        x = x.view(-1, 3, 3, self.class_num, t)

        return x
