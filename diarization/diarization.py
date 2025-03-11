import torch
from pyannote.audio import Model, Pipeline
from pyannote.database.util import load_rttm
import os
from pathlib import Path
from config_loader import config


class diarization:
    def __init__(self, device):
        
        self.device = device

        self.pipeline = Pipeline.from_pretrained(config['Paths']['PYANNOTE_CONFIG']).to(device)
        self.sample_rate = int(config['Params']['SR'])

    def process(self, file_path, mono_audio):
        diarization = self.pipeline({
            "waveform": mono_audio.to(self.device),
            "sample_rate": self.sample_rate
        }, min_speakers=1, max_speakers=4)

        diarization.uri = os.path.splitext(os.path.basename(file_path))[0]

        return diarization

    def save_rttm(self, file_path, diarization, saved_dir='rttm'):

        corpus_path = os.path.dirname(os.path.dirname(file_path))
        output_dir = os.path.join(corpus_path, saved_dir)
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        file_name = os.path.basename(file_path).replace('.wav', '.rttm')
        output_file = os.path.join(output_dir, file_name)

        with open(output_file,'w') as f:
            diarization.write_rttm(f)

        return output_file

    def empty_cache(self):
        del self.pipeline
        torch.cuda.empty_cache()


        