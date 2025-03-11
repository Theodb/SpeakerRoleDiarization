#!/usr/bin/env python
import argparse
import os
import glob
import json
import time
import gc
import logging
import configparser
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

# Import custom modules
#import demucs.separate  # Implements the Demucs model
from preprocessing.preprocess_audio import preprocess_audio
from preprocessing.seperate_channels import seperate_channels
from label_studio.push_to_label_studio import save_to_json_for_humanlabel
from speech_to_text.ASR import ASR
from voice_activity.VAD import VAD
from alignment.CTC import CTC
from alignment.CTC_v2 import load_align_model, align, AlignedTranscriptionResult
from diarization.diarization import diarization
from utils.align_words_with_speakers import align_words_with_speakers
from utils.utils import *

# Configure logging
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("speechbrain").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Global config variables (updated from CLI arguments in main)
G_DEVICE = None
LANG = "fr"
ASR_MODEL = "large-v3"
BACKEND = "faster-whisper"
BEAM_SIZE = 5

@contextmanager
def load_vad_model():
    model = VAD(sample_rate=16000)
    try:
        yield model
    finally:
        model.empty_cache()

@contextmanager
def load_ctc_model():
    model = CTC(G_DEVICE, language=LANG)
    try:
        yield model
    finally:
        model.empty_cache()

@contextmanager
def init_asr():
    asr_instance = ASR(
        device=G_DEVICE,
        compute_type='float16',
        asr_model=ASR_MODEL,
        backend=BACKEND,
        beam_size=BEAM_SIZE
    )
    try:
        yield asr_instance
    finally:
        # Call any cleanup methods if available.
        pass

@contextmanager
def load_diarization_model():
    model = diarization(G_DEVICE)
    try:
        yield model
    finally:
        model.empty_cache()

def process_option2(stereo_path, client_waveform, agent_waveform, client_audio_path, agent_audio_path):
    """
    Option 2: Use VAD to extract speech, then ASR and CTC for alignment.
    """
    try:
        with load_vad_model() as vad_model:
            client_timestamps, only_speech_client = vad_model.process(client_waveform, silence_duration=0)
            agent_timestamps, only_speech_agent = vad_model.process(agent_waveform, silence_duration=0)
            only_speech_path_client = vad_model.save_audio(stereo_path, only_speech_client, saved_dir='speech_only_client')
            only_speech_path_agent = vad_model.save_audio(stereo_path, only_speech_agent, saved_dir='speech_only_agent')

        # Save RTTM file (oracle)
        oracle_rttm_path = vad_model.save_rttm(stereo_path, client_timestamps, agent_timestamps, saved_dir='oracle_rttm_option2')

        with init_asr() as asr_instance:
            trs_client = asr_instance.process(only_speech_path_client)
            trs_agent = asr_instance.process(only_speech_path_agent)

        with load_ctc_model() as ctc_model:
            alignments_client = ctc_model.process(client_waveform, ' '.join(pd.DataFrame(trs_client)['text']))
            alignments_agent = ctc_model.process(agent_waveform, ' '.join(pd.DataFrame(trs_agent)['text']))
            words_csv_path_client = ctc_model.save(stereo_path, alignments_client, saved_dir='auto_trs_aligned_client')
            words_csv_path_agent = ctc_model.save(stereo_path, alignments_agent, saved_dir='auto_trs_aligned_agent')

        csv_client = pd.read_csv(words_csv_path_client)
        csv_agent = pd.read_csv(words_csv_path_agent)
        csv_client['speaker'] = 'client'
        csv_agent['speaker'] = 'agent'
        csv_combined = pd.concat([csv_client, csv_agent], ignore_index=True)
        csv_combined.sort_values(by=['start'], inplace=True)
        csv_combined['text'] = csv_combined['text'].fillna('')

        align_words_with_speakers(csv_combined, oracle_rttm_path, stereo_path, saved_dir='auto_trs_aligned_speakers_oracle')
    except Exception as e:
        logging.error(f"Error processing option 2 for file {stereo_path}: {e}")

def process_option3(stereo_path, client_audio_path, agent_audio_path):
    """
    Option 3: Use VAD + ASR, then apply alignment via CTC_v2.
    """
    try:
        with init_asr() as asr_instance:
            trs_client = asr_instance.process(client_audio_path)
            trs_agent = asr_instance.process(agent_audio_path)

        segments_client = [{'start': seg['start'], 'end': seg['end'], 'text': seg['text']} for seg in trs_client]
        segments_agent = [{'start': seg['start'], 'end': seg['end'], 'text': seg['text']} for seg in trs_agent]

        interpolate_method = "nearest"
        return_char_alignments = False
        print_progress = False

        # Load alignment model and metadata
        align_model, align_metadata = load_align_model(LANG, G_DEVICE, model_name=None)
        logging.info("Performing alignment for client...")
        aligned_result_client = align(
            transcript=segments_client,
            model=align_model,
            align_model_metadata=align_metadata,
            audio=client_audio_path,
            device=G_DEVICE,
            interpolate_method=interpolate_method,
            return_char_alignments=return_char_alignments,
            print_progress=print_progress,
        )
        logging.info("Performing alignment for agent...")
        aligned_result_agent = align(
            transcript=segments_agent,
            model=align_model,
            align_model_metadata=align_metadata,
            audio=agent_audio_path,
            device=G_DEVICE,
            interpolate_method=interpolate_method,
            return_char_alignments=return_char_alignments,
            print_progress=print_progress,
        )
        # Clean up alignment model
        del align_model
        gc.collect()
        torch.cuda.empty_cache()

        # Merge aligned segments and sort them
        merged_segments = []
        for segment in aligned_result_agent['segments']:
            merged_segments.append({'start': segment['start'], 'end': segment['end'],
                                    'text': segment['text'], 'speaker': 'agent'})
        for segment in aligned_result_client['segments']:
            merged_segments.append({'start': segment['start'], 'end': segment['end'],
                                    'text': segment['text'], 'speaker': 'client'})
        merged_segments.sort(key=lambda x: x['start'])
        segments_df = pd.DataFrame(merged_segments)

        # Postprocessing: remove known hallucinations
        hallucinations = [
            'Sous-titrage ST',
            'Sous-titrage Société Radio-Canada',
            'Sous-titrage FR ?',
            "' 501",
            '- Sous-titrage FR 2021',
            '– Sous-titrage FR 2021',
            "Sous-titrage ST' 501",
            '- 2021',
            '– 2021'
        ]
        segments_df['text'] = segments_df['text'].replace(hallucinations, '', regex=True)
        segments_df['text'] = segments_df['text'].fillna('')

        corpus_path = os.path.dirname(os.path.dirname(stereo_path))
        output_dir = os.path.join(corpus_path, 'trs_per_segments')
        os.makedirs(output_dir, exist_ok=True)
        segment_df_path = os.path.join(output_dir, os.path.splitext(os.path.basename(stereo_path))[0] + '.csv')
        segments_df.to_csv(segment_df_path, index=False)

        # Process word-level alignments
        merged_words = []
        for segment in aligned_result_agent['segments']:
            for word in segment.get('words', []):
                merged_words.append({'start': word['start'], 'end': word['end'],
                                     'word': word['word'], 'speaker': 'agent'})
        for segment in aligned_result_client['segments']:
            for word in segment.get('words', []):
                merged_words.append({'start': word['start'], 'end': word['end'],
                                     'word': word['word'], 'speaker': 'client'})
        merged_words.sort(key=lambda x: x['start'])
        words_df = pd.DataFrame(merged_words)
        output_dir = os.path.join(corpus_path, 'trs_per_words')
        os.makedirs(output_dir, exist_ok=True)
        word_df_path = os.path.join(output_dir, os.path.splitext(os.path.basename(stereo_path))[0] + '.csv')
        words_df['text'] = words_df.get('text', '').fillna('')
        words_df.to_csv(word_df_path, index=False)

        # Optionally, push to Label Studio for human labeling
        save_to_json_for_humanlabel(segment_df_path, stereo_path, None,
                                    bucket_name="theo-deschamps-test", saved_dir="json_segs_LabelStudio")
    except Exception as e:
        logging.error(f"Error processing option 3 for file {stereo_path}: {e}")

def process_option4(stereo_path, client_waveform, agent_waveform, client_audio_path, agent_audio_path, use_parallel, use_context, sample_rate=16000):
    """
    Option 4: Use VAD to segment, then ASR on segments with optional rolling context.
    """
    #try:
    with load_vad_model() as vad_model:
        client_timestamps, _ = vad_model.process(client_waveform, silence_duration=0)
        agent_timestamps, _ = vad_model.process(agent_waveform, silence_duration=0)
        #_ = vad_model.save_audio(stereo_path, only_speech_client, saved_dir='speech_only_client')  # Optional saving
        #_ = vad_model.save_audio(stereo_path, only_speech_agent, saved_dir='speech_only_agent')

    def limit_rolling_context(rolling_context, max_chars=200):
        return rolling_context[-max_chars:] if len(rolling_context) > max_chars else rolling_context

    def transcribe_segment(asr_instance, waveform, seg, role, rolling_context):
        start_idx = int(seg["start"] * sample_rate)
        end_idx = int(seg["end"] * sample_rate)
        audio_segment = waveform[:, start_idx:end_idx]
        if use_context:
            rolling_context = limit_rolling_context(rolling_context)
        segment_trs = asr_instance.process(audio_segment, rolling_context=rolling_context, vad_filter=False)
        final_trs = ' '.join([s["text"] for s in segment_trs])
        if use_context:
            rolling_context += " " + final_trs
        return {"start": seg["start"], "end": seg["end"], "text": final_trs, "speaker": role}, rolling_context

    def process_segments(waveform, timestamps, role, use_parallel):
        processed_results = []
        rolling_context = ""
        if use_parallel:
            with init_asr() as asr_instance, ThreadPoolExecutor() as executor:
                futures = {executor.submit(transcribe_segment, asr_instance, waveform, seg, role, rolling_context): seg 
                            for seg in timestamps}
                for future in futures:
                    result, _ = future.result()
                    processed_results.append(result)
        else:
            with init_asr() as asr_instance:
                for seg in timestamps:
                    result, rolling_context = transcribe_segment(asr_instance, waveform, seg, role, rolling_context)
                    processed_results.append(result)
        return processed_results

    client_results = process_segments(client_waveform, client_timestamps, "client", use_parallel)
    agent_results = process_segments(agent_waveform, agent_timestamps, "agent", use_parallel)
    all_segments = client_results + agent_results
    all_segments.sort(key=lambda x: x["start"])
    segments_df = pd.DataFrame(all_segments)

    hallucinations = [
        'Sous-titrage ST',
        'Sous-titrage Société Radio-Canada',
        'Sous-titrage FR ?',
        "' 501",
        '- Sous-titrage FR 2021',
        '– Sous-titrage FR 2021',
        "Sous-titrage ST' 501",
        '- 2021',
        '– 2021'
    ]
    segments_df['text'] = segments_df['text'].replace(hallucinations, '', regex=True)
    segments_df['text'] = segments_df['text'].fillna('')

    corpus_path = os.path.dirname(os.path.dirname(stereo_path))
    output_dir = os.path.join(corpus_path, 'trs_per_segments_option4')
    os.makedirs(output_dir, exist_ok=True)
    segment_df_path = os.path.join(output_dir, os.path.splitext(os.path.basename(stereo_path))[0] + '.csv')
    segments_df.to_csv(segment_df_path, index=False)
    #except Exception as e:
    #    logging.error(f"Error processing option 4 for file {stereo_path}: {e}")

def process_audio_file(stereo_path, option, use_parallel, use_context):
    # Preprocess and load audio file
    processed_path = preprocess_audio(stereo_path, saved_dir='audio_wav')
    waveform, sample_rate = torchaudio.load(processed_path)
    length_audio = waveform.size(1) / sample_rate
    logging.info(f"Processing file {processed_path} ({length_audio:.2f} sec)")

    # Only process stereo files
    if waveform.ndim == 2 and waveform.size(0) == 2:
        client_audio_path, agent_audio_path = seperate_channels(processed_path)
        client_waveform, _ = torchaudio.load(client_audio_path)
        agent_waveform, _ = torchaudio.load(agent_audio_path)

        if option == 1:
            process_option1(processed_path, client_audio_path, agent_audio_path)
        elif option == 2:
            process_option2(processed_path, client_waveform, agent_waveform, client_audio_path, agent_audio_path)
        elif option == 3:
            process_option3(processed_path, client_audio_path, agent_audio_path)
        elif option == 4:
            process_option4(processed_path, client_waveform, agent_waveform,
                            client_audio_path, agent_audio_path, use_parallel, use_context, sample_rate)
    else:
        logging.warning(f"File {processed_path} is not stereo; skipping.")

def main():
    parser = argparse.ArgumentParser(
        description="Audio processing pipeline with multiple transcription and alignment options."
    )
    parser.add_argument("--input_dir", type=str, default="/root/theo_db/projets/DiarAndIdentify/data/audio",
                        help="Directory containing input audio files.")
    parser.add_argument("--duplicates_file", type=str, default="doublons.json",
                        help="Path to JSON file with duplicate conversation info.")
    parser.add_argument("--option", type=int, choices=[1, 2, 3, 4], default=4,
                        help="Processing option: "
                             "1 = ASR only, "
                             "2 = VAD + ASR + CTC, "
                             "3 = VAD + ASR + Alignment, "
                             "4 = VAD + ASR with context.")
    parser.add_argument("--max_files", type=int, default=1,
                        help="Maximum number of files to process (-1 for no limit).")
    parser.add_argument("--parallel", dest="parallel", action="store_true", default=True,
                        help="Enable parallel processing (option 4).")
    parser.add_argument("--no_parallel", dest="parallel", action="store_false",
                        help="Disable parallel processing (option 4).")
    parser.add_argument("--use_context", action="store_true", default=False,
                        help="Use rolling context during transcription (option 4).")
    parser.add_argument("--language", type=str, default="fr",
                        help="Language code for transcription and alignment.")
    parser.add_argument("--asr_model", type=str, default="large-v3",
                        help="ASR model to use.")
    parser.add_argument("--backend", type=str, default="faster-whisper",
                        help="ASR backend to use.")
    parser.add_argument("--beam_size", type=int, default=5,
                        help="Beam size for ASR decoding.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"],
                        help="Force device (default auto-detect).")
    args = parser.parse_args()

    # Update global settings
    global G_DEVICE, LANG, ASR_MODEL, BACKEND, BEAM_SIZE
    LANG = args.language
    ASR_MODEL = args.asr_model
    BACKEND = args.backend
    BEAM_SIZE = args.beam_size

    if args.device:
        G_DEVICE = torch.device(args.device)
    else:
        if torch.backends.mps.is_available():
            logging.info("MPS is not compatible, switching to CPU")
            G_DEVICE = torch.device("cpu")
        elif torch.cuda.is_available():
            G_DEVICE = torch.device("cuda")
        else:
            G_DEVICE = torch.device("cpu")
            logging.info("Neither CUDA nor MPS available. Using CPU.")
    logging.info(f"Using device: {G_DEVICE}")

    # Load configuration if needed
    script_path = os.path.abspath(os.path.dirname(__file__))
    config = configparser.ConfigParser()
    config.read(os.path.join(script_path, 'config.ini'))

    # Retrieve audio files and filter duplicates
    files = sorted(glob.glob(os.path.join(args.input_dir, '*')))
    duplicates_file_path = args.duplicates_file
    if not os.path.isabs(duplicates_file_path):
        duplicates_file_path = os.path.join(script_path, duplicates_file_path)
    with open(duplicates_file_path, 'r') as fp:
        conv_duplicates = json.load(fp)
    duplicate_files = []
    for k, v in conv_duplicates.items():
        duplicate_files.extend(v[1:])
    duplicate_files = [os.path.splitext(os.path.basename(f))[0] for f in duplicate_files]
    files = [f for f in files if os.path.splitext(os.path.basename(f))[0] not in duplicate_files]
    if args.max_files > 0:
        files = files[:args.max_files]

    start_time = time.time()
    for stereo_path in tqdm(files, desc="Processing files"):
        process_audio_file(stereo_path, args.option, args.parallel, args.use_context)
    end_time = time.time()
    logging.info(f"Script executed in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
