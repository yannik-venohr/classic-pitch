import torch
import torch.nn as nn
from src.model.onset import  OnsetModule
from src.model.hcqt_encoder import HCQTEncoder


class Assembler(nn.Module):
    def __init__(self, model_params: dict) -> None:
        super().__init__()
        self.log_compress_input = model_params.get("log_compress_input")
        self.assembly_method = model_params["assembly_method"]
        if model_params["mpe_encoder"] is not None:
            cfg = model_params["mpe_encoder"]
            self.mpe_encoder = HCQTEncoder(**cfg, pitch_class_head=model_params.get("pitch_class_head", False))

        if model_params.get("onset_post") is not None:
            cfg = model_params["onset_post"]
            self.onset_post = OnsetModule(**cfg)

        self.use_pitch_class_head = model_params.get("pitch_class_head", False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.log_compress_input:
            x = torch.log1p(x * 1000)

        if self.assembly_method == "just-mpe":
            x_note = self.mpe_encoder(x) # (B, 1, F, T)
            x_onset = x_note.clone() # (B, 1, F, T)
        elif self.assembly_method == "classic_pitch":
            features = self.mpe_encoder.forward_features(x)  # (B, 80, 72, T)
            x_note = self.mpe_encoder.head(features)          # (B, 1, 72, T)
            x_onset = self.onset_post(features)               # (B, 1, 72, T)

        out = {
            "note": x_note.squeeze(1),   # (B, F, T)
            "onset": x_onset.squeeze(1), # (B, F, T)
        }
        if self.use_pitch_class_head:
            out["pitch_class"] = self.mpe_encoder.pitch_class_head(torch.sigmoid(x_note)).squeeze(1)
        return out

