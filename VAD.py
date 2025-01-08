from silero_vad import load_silero_vad, get_speech_timestamps
import soundfile as sf
import numpy as np
import torch
import os
from process_rttm import LabelRTTM

class VAD:
    """
    Voice Activity Detection (VAD) class using the Silero VAD model.

    This class processes audio files to isolate speech segments and provides functionality 
    to save the processed speech audio to a file.

    Attributes:
        model (torch.nn.Module): Preloaded Silero VAD model for detecting speech.
        sample_rate (int): Sampling rate of the input audio.
    """
    def __init__(self, sample_rate):
        self.model = load_silero_vad()
        self.sample_rate = sample_rate

    def process(self, waveform):
        #returns audio with only speech

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
        speech_audio = []
        for segment in speech_timestamps:
            start, end = segment['start'], segment['end']
            start_sample = int(start * self.sample_rate)
            end_sample = int(end * self.sample_rate)
            speech_audio.append(waveform[start_sample:end_sample])

        # Combine all speech segments into one audio stream
        if speech_audio:
            merged_audio = np.concatenate(speech_audio)
        else:
            raise ValueError("No speech detected in the audio file.")

        return speech_timestamps, merged_audio

    def save_audio(self, file_path, merged_audio, saved_dir='speech_only'):
        corpus_path = os.path.dirname(os.path.dirname(file_path))
        output_dir = os.path.join(corpus_path, saved_dir)
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        file_name = os.path.basename(file_path)
        output_file = os.path.join(output_dir, file_name)

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
        f.write(''.join([LabelRTTM(os.path.splitext(file_name)[0], i['start'], (i['end'] - i['start']), i['name']).format_rttm() for i in speakers]))
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
