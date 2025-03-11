import subprocess
import os
import torch
import numpy as np
from config_loader import config
from faster_whisper import WhisperModel, BatchedInferencePipeline

class ASR:
    def __init__(self, device='cpu', compute_type='float32', asr_model="large-v3", backend="faster-whisper", beam_size=5):
        
        models_path = config["PATHS"]["models"]
        self.backend = backend
        self.sample_rate = 16000
        self.beam_size = beam_size
        self.batch_size = 8

        if backend == "faster-whisper":
            model_path = os.path.join(models_path, f"faster-whisper-{asr_model}")
            self.model = WhisperModel(model_path, device="cuda", compute_type="float16")  # Adjust device as needed
            #self.model = BatchedInferencePipeline(model=model)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def process(self, input_data, rolling_context=None, vad_filter=True):
        """
        Process an audio file or waveform with optional rolling context.

        Args:
            input_data (str or torch.Tensor): Path to the audio file or audio waveform (Tensor).
            rolling_context (str): Previous transcription context.

        Returns:
            List[dict]: Transcription with timestamps and text segments.
        """
        if isinstance(input_data, str):  # File path
            file_path = input_data
        elif isinstance(input_data, torch.Tensor):  # Audio waveform
            # Convert PyTorch tensor to NumPy array
            waveform = input_data.detach().cpu().numpy()
            # Ensure the waveform is in the correct shape (1D array)
            if waveform.ndim > 1:
                waveform = waveform.squeeze()
            # Normalize the waveform if it's in integer format
            if waveform.dtype == np.int16:
                waveform = waveform.astype(np.float32) / 32768.0
            file_path = waveform

        if self.backend == "whispercpp":
            #-nt without timestamps
            full_command = f"{self.whisper_cpp_path} -m {self.model} -f {file_path} --language fr"
            query = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = query.communicate()
            #print(output)
            #print(error)
            # Process and return the output string
            decoded_str = output.decode('utf-8').strip()
            processed_str = decoded_str.replace('[BLANK_AUDIO]', '').strip()

            return processed_str
        elif self.backend == "faster-whisper":
            #condition_on_previous_text: If default(True), the previous output of the model is provided as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.
            # repetition_penalty: default 1 Penalty applied to the score of previously generated tokens (set > 1 to penalize).
            # no_repeat_ngram_size: default 0 Prevent repetitions of ngrams with this size (set 0 to disable).

            segments, _ = self.model.transcribe(file_path, beam_size = self.beam_size, language = 'fr', condition_on_previous_text=False, vad_filter=False, repetition_penalty = 1.25, no_repeat_ngram_size=0, without_timestamps=True)
            #segments, _ = self.model.transcribe(
            #    file_path, 
            #    language = 'fr', 
            #    vad_filter=False, 
            #    condition_on_previous_text=False, 
            #    repetition_penalty = 1.25, 
            #    no_repeat_ngram_size=0, 
            #    without_timestamps=True, 
            #    temperature=0.0,
            #    beam_size= 5,
            #    best_of= 5,
            #    patience= 1,
            #    length_penalty= 1,
            #    compression_ratio_threshold= 2.4,
            #    log_prob_threshold= -1.0,
            #    no_speech_threshold= 0.6,
            #    prompt_reset_on_temperature= 0.5,
            #    initial_prompt= None,
            #    prefix= None,
            #    suppress_blank= True,
            #    suppress_tokens= [-1],
            #    max_initial_timestamp= 0.0,
            #    word_timestamps= False,
            #    prepend_punctuations= "\"'“¿([{-",
            #    append_punctuations= "\"'.。,，!！?？:：”)]}、",
            #    max_new_tokens= None,
            #    clip_timestamps= None,
            #    hallucination_silence_threshold= None,
            #    hotwords= None,
            #    multilingual= True,
            #    )#, #initial_prompt="Biovancia")
    
            sentences = [{"start": segment.start, "end": segment.end, "text": segment.text.strip()} for segment in segments]
            return sentences

    def save(self, file_path, processed_str, saved_dir='trs'):
        corpus_path = os.path.dirname(os.path.dirname(file_path))
        output_dir = os.path.join(corpus_path, saved_dir)
        os.makedirs(output_dir, exist_ok=True)
        file_name = os.path.splitext(os.path.basename(file_path))[0] + '.txt'
        output_file = os.path.join(output_dir, file_name)

        try:
            with open(output_file, 'w') as f:
                f.write(processed_str)
        except IOError as e:
            print(f"Error saving transcription: {e}")
