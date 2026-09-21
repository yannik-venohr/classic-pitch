from predictor import ClassicPitch
from pathlib import Path
from tqdm import tqdm
import numpy as np

audio_dir = Path('data/')
output_dir = Path('data/predictions/')
output_dir.mkdir(parents=True, exist_ok=True)

predictor = ClassicPitch(device='mps', merge_notes=False)
for wav_file in tqdm(audio_dir.glob('*.wav'), desc='Processing audio files', total=len(list(audio_dir.glob('*.wav')))):
    # check if the output files already exist
    if (output_dir / f'{wav_file.stem}_frame_activation.npy').exists() and (output_dir / f'{wav_file.stem}.mid').exists():
        continue
    id = wav_file.stem
    midi, model_output, times = predictor.predict(wav_file)
    frame_activation = model_output['note']
    np.save(output_dir / f'{id}_frame_activation.npy', frame_activation)
    midi.write(output_dir / f'{id}.mid')
