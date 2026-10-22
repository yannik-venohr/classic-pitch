import torch
import torch.nn as nn
from src.model.hcqt_encoder import HCQTEncoder
from src.model.onset import OnsetModule


class Assembler(nn.Module):
    """
    Builds the ClassicPitch model from the model config: HCQTEncoder for frames,
    OnsetModule on its features for onsets.
    Input:  HCQT (B, C, F, T)
    Output: dict with 'note' and 'onset' logits (B, 72, T)
    """
    def __init__(self, model_params: dict) -> None:
        super().__init__()
        self.log_compress_input = model_params.get("log_compress_input")
        self.assembly_method = model_params["assembly_method"]
        if self.assembly_method != "classic_pitch":
            raise ValueError(f"Unsupported assembly_method: {self.assembly_method}")

        self.mpe_encoder = HCQTEncoder(**model_params["mpe_encoder"])
        self.onset_post = OnsetModule(**model_params["onset_post"])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.log_compress_input:
            x = torch.log1p(x * 1000)

        features = self.mpe_encoder.forward_features(x)  # (B, C, F, T)
        x_note = self.mpe_encoder.head(features)         # (B, 1, F, T)
        x_onset = self.onset_post(features)              # (B, 1, F, T)

        return {
            "note": x_note.squeeze(1),   # (B, F, T)
            "onset": x_onset.squeeze(1), # (B, F, T)
        }
