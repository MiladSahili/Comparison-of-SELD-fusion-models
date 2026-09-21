import json
import torch
import torch.nn as nn

from src.models.audioonly.net_seld import AudioOnlyCRNN
from src.models.visual_only_2.net_seld import VisualOnlyCRNN_2


def create_net_seld(args):
    with open(args.feature_config, 'r') as f:
        feature_config = json.load(f)

    alpha = getattr(args, 'late_fusion_alpha', 0.5)

    Net = LateFusionShuffledCRNN(
        class_num=args.class_num,
        in_channels=feature_config[args.feature]["ch"],
        alpha=alpha,
    )
    return Net


class LateFusionShuffledCRNN(nn.Module):
    """Late-Fusion-Kontrollexperiment mit geshuffelten visuellen Features.

    Identisch zu LateFusionCRNN, aber die visuellen Features werden vor
    jedem Forward ueber die Batch-Dimension permutiert. Audio- und
    Visual-Eingang eines Samples korrespondieren dann nicht mehr –
    das Modell kann die visuelle Information nicht mehr sinnvoll nutzen.

    Zweck: Negativ-Kontrolle. Liegt das Ergebnis auf Audio-Only-Niveau,
    nutzt das ungeschuffelte Late-Fusion-Modell die visuelle Information
    tatsaechlich; liegt es darunter, stammt der Nutzen aus der korrekten
    Audio-Visual-Korrespondenz.

    Das Shuffeln passiert IMMER (auch im Eval-Modus), damit die Validierung
    dieselbe Bedingung sieht wie das Training.
    """

    def __init__(self, class_num, in_channels, alpha=0.5):
        super().__init__()
        self.class_num = class_num
        self.alpha = alpha

        self.audio_branch = AudioOnlyCRNN(class_num=class_num, in_channels=in_channels)
        self.visual_branch = VisualOnlyCRNN_2(class_num=class_num)

    def forward(self, x_a, x_v):
        # Visuelle Features ueber die Batch-Dimension shuffeln:
        # Sample i bekommt das Visual-Feature von Sample perm[i].
        idx = torch.randperm(x_v.size(0), device=x_v.device)
        x_v_shuffled = x_v[idx]

        out_a = self.audio_branch(x_a, x_v_shuffled)
        out_v = self.visual_branch(x_a, x_v_shuffled)

        out = self.alpha * out_a + (1.0 - self.alpha) * out_v

        if self.training:
            return out, out_a, out_v
        return out
