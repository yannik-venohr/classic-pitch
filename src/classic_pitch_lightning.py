import torch
from lightning import LightningModule
from src.model.assembler import Assembler


class ClassicPitchLightning(LightningModule):
    def __init__(self, model_config, training_config):
        """
        Initialize the ClassicPitchLightning module.
        Args:
            model_config (dict): Configuration for the model.
            training_config (dict): Configuration for training parameters.
        """
        super().__init__()
        self.model = Assembler(model_config)
        self.loss_config = training_config.loss
        self.learning_rate = training_config.learning_rate
        self.max_steps = training_config.max_steps
        self.save_hyperparameters()

    def predict_full_track(self, hcqt, max_frames=1000, overlap=10, batch_size=32):
        hcqt = torch.as_tensor(hcqt, dtype=torch.float32, device=self.device)

        T = hcqt.shape[-1]
        step = max_frames - overlap

        # 1. Collect chunk boundaries
        starts = list(range(0, T, step))

        # 2. Pad input so all chunks are max_frames long
        pad_needed = max(0, starts[-1] + max_frames - T)
        if pad_needed:
            hcqt = torch.nn.functional.pad(hcqt, (0, pad_needed))

        # 3. Stack all chunks → (N, C, F, max_frames)
        chunks = torch.stack([hcqt[..., s:s + max_frames] for s in starts])

        # 4. Batched inference
        self.model.eval()
        all_outputs = {}
        with torch.no_grad():
            for i in range(0, len(chunks), batch_size):
                out = self.model(chunks[i:i + batch_size])
                for k, v in out.items():
                    all_outputs.setdefault(k, []).append(v.cpu())
        all_outputs = {k: torch.cat(v, dim=0) for k, v in all_outputs.items()}

        # 5. Stitch: trim overlap and reassemble
        results = {}
        for k, v in all_outputs.items():
            pieces = []
            for j, s in enumerate(starts):
                chunk_out = v[j]
                if j > 0:
                    chunk_out = chunk_out[..., overlap:]
                pieces.append(chunk_out)
            full = torch.cat(pieces, dim=-1)[..., :T]
            results[k] = torch.sigmoid(full).numpy()
        return results

