from silero_vad import load_silero_vad, get_speech_timestamps
import soundfile as sf
import numpy as np
import torch
import os
from utils.process_rttm import LabelRTTM
from pyannote.audio.pipelines import VoiceActivityDetection as VoiceActivityDetectionPipeline
from pyannote.audio import Model
from config_loader import config

class VAD:
    """
    Voice Activity Detection (VAD) class using the Silero VAD model.

    This class processes audio files to isolate speech segments and provides functionality 
    to save the processed speech audio to a file.

    Attributes:
        model (torch.nn.Module): Preloaded Silero VAD model for detecting speech.
        sample_rate (int): Sampling rate of the input audio.
    """
    def __init__(self, sample_rate, model_name='pyannote/segmentation-3.0'):

        self.model_name = model_name

        if 'pyannote' in self.model_name:
            initial_params = {"min_duration_on": 0.3, "min_duration_off": 0.3} #{"onset": 0.500, "offset": 0.363,

            vad_model = Model.from_pretrained(
                model_name,
                use_auth_token=config['TOKENS']['hf'])
            vad_pipe = VoiceActivityDetectionPipeline(segmentation=vad_model)
            self.model = vad_pipe.instantiate(initial_params)
        else:
            self.model = load_silero_vad()

        self.sample_rate = sample_rate

    def process(self, waveform, silence_duration=0):
        #returns audio with only speech
        
        #min_speech_duration_ms = 600
        #min_silence_duration_ms = 200
        #with torch.no_grad():
        #    # Get speech timestamps
        #    speech_timestamps = get_speech_timestamps(
        #        audio=waveform,
        #        model=self.model,
        #        threshold=0.25,
        #        sampling_rate=16000,
        #        min_speech_duration_ms=min_speech_duration_ms,
        #        max_speech_duration_s=float("inf"),
        #        min_silence_duration_ms=min_silence_duration_ms,
        #        speech_pad_ms=200,
        #        return_seconds=False,
        #        visualize_probs=False,
        #        progress_tracking_callback=None,
        #        neg_threshold=None,
        #        window_size_samples=512,
        #        )

        if 'pyannote' in self.model_name:
            # Process the audio with the model
            vad_result = self.model({'waveform': waveform, 'sample_rate': self.sample_rate})
            speech_timestamps = [{'start': segment.start, 'end': segment.end} for segment in vad_result.get_timeline()]

        elif 'silero' in self.model_name:
            # Check if the waveform needs squeezing
            if waveform.ndim > 1 and waveform.shape[0] == 1:
                waveform = waveform.squeeze(0)  # Remove the first dimension if it is singleton

            with torch.no_grad():
                # Get speech timestamps
                speech_timestamps = get_speech_timestamps(
                    waveform,
                    self.model,
                    return_seconds=True  # Return speech timestamps in seconds (default is samples)
                )
                
        # Merge audio segments corresponding to speech timestamps
        # Assuming waveform is a torch.Tensor of shape (1, 1043840)

        silence_samples = int(silence_duration * 16000)  # Convert silence duration to sample count
        silence = torch.zeros((1, silence_samples))  # Create a silence tensor

        speech_audio = []
        for i, segment in enumerate(speech_timestamps):
            start, end = segment['start'], segment['end']
            start_sample = int(start * 16000)
            end_sample = int(end * 16000)
            # Slice the waveform while keeping the channel dimension
            speech_audio.append(waveform[:, start_sample:end_sample])
            # Add silence between segments, except after the last segment
            if i < len(speech_timestamps) - 1:
                speech_audio.append(silence)

        # Combine all speech segments into one audio stream as a single PyTorch tensor
        merged_audio = torch.cat(speech_audio, dim=1)  # Concatenate along the time dimension

        return speech_timestamps, merged_audio

    def save_audio(self, file_path, merged_audio, saved_dir='speech_only'):
        corpus_path = os.path.dirname(os.path.dirname(file_path))
        output_dir = os.path.join(corpus_path, saved_dir)
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        file_name = os.path.basename(file_path)
        output_file = os.path.join(output_dir, file_name)

        # Convert PyTorch tensor to NumPy array before saving
        if torch.is_tensor(merged_audio):
            # Squeeze the channel dimension (if present) for saving as mono audio
            merged_audio = merged_audio.squeeze(0).cpu().numpy()

        sf.write(output_file, merged_audio, self.sample_rate)

        return output_file

    def save_rttm(self, file_path, client_timestamps, agent_timestamps, saved_dir='oracle_rttm'):

        for entry in client_timestamps:
            entry['name']='client'
        for entry in agent_timestamps:
            entry['name']='agent'

        corpus_path = os.path.dirname(os.path.dirname(file_path))
        output_dir = os.path.join(corpus_path, saved_dir)
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        file_name = os.path.basename(file_path).replace('.wav', '.rttm')
        output_file = os.path.join(output_dir, file_name)

        # get process_rttm file from https://gist.github.com/pthavarasa/f873ad3cdd3a6c9fef7122eb1ae12dd4

        speakers = sorted(client_timestamps + agent_timestamps, key=lambda x: x['start'])
        f = open(output_file, 'w')
        f.write(''.join([LabelRTTM(os.path.splitext(file_name)[0], i['start'], ((i['end'] - i['start'])), i['name']).format_rttm() for i in speakers]))
        f.close()
        # Open the output file
        #with open(output_file, 'w') as f:
        #    for speaker in speakers:
        #        start_time = round(speaker['start'] / self.sample_rate, 6)
        #        duration = round((speaker['end'] - speaker['start']) / self.sample_rate, 6)
        #        name = speaker['name']
#
        #        # Skip negligible durations
        #        if duration < 1e-4:
        #            continue
#
        #        # Create and format the RTTM line
        #        rttm_line = LabelRTTM(
        #            fileName=os.path.splitext(file_name)[0],
        #            startTime=start_time,
        #            duration=duration,
        #            speakerName=name
        #        ).format_rttm()
        #        # Write the RTTM line
        #        f.write(rttm_line)

        return output_file

    def empty_cache(self):
        del self.model
        torch.cuda.empty_cache()
