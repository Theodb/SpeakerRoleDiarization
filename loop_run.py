#!pip install ctranslate2==4.4.0

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm 
import glob
import logging
# Set the logging level for the faster_whisper logger to WARNING or ERROR
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("speechbrain").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
import json

#INFO:speechbrain.utils.quirks:Applied quirks (see `speechbrain.utils.quirks`): [disable_jit_profiling, allow_tf32]
#INFO:speechbrain.utils.quirks:Excluded quirks specified by the `SB_DISABLE_QUIRKS` environment (comma-separated list): []
#INFO:datasets:PyTorch version 2.5.1 available.

import demucs.separate #implements the Demucs model

from preprocess_audio import preprocess_audio
from seperate_channels import seperate_channels
from push_to_label_studio import save_to_json_for_humanlabel

from ASR import ASR
from VAD import VAD
from CTC import CTC
from diarization import diarization

from align_words_with_speakers import align_words_with_speakers

import glob
import os
import pandas as pd
from contextlib import contextmanager
import torch
import torchaudio

import gc

# Check device compatibility
if torch.backends.mps.is_available():
    print("MPS is still not compatible switching to cpu")
    device = torch.device("cpu")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    compute_type = torch.float16
else:
    device = torch.device("cpu")
    print("Neither MPS nor CUDA is available. Using CPU.")
print(f"Using device: {device}")

def to_timestamp(t: int, separator=',') -> str:
    """
    376 -> 00:00:03,760
    1344 -> 00:00:13,440

    Implementation from `whisper.cpp/examples/main`

    :param t: input time from whisper timestamps
    :param separator: seprator between seconds and milliseconds
    :return: time representation in hh: mm: ss[separator]ms
    """
    # logic exactly from whisper.cpp

    msec = t * 10
    hr = msec // (1000 * 60 * 60)
    msec = msec - hr * (1000 * 60 * 60)
    min = msec // (1000 * 60)
    msec = msec - min * (1000 * 60)
    sec = msec // 1000
    msec = msec - sec * 1000
    return f"{int(hr):02,.0f}:{int(min):02,.0f}:{int(sec):02,.0f}{separator}{int(msec):03,.0f}"

def to_seconds(t: int) -> float:
    """
    Convert a given timestamp (in Whisper's format) into seconds.

    Implementation from `whisper.cpp/examples/main`

    :param t: input time from whisper timestamps
    :return: time in seconds
    """
    # Convert the input timestamp to milliseconds
    msec = t * 10
    
    # Convert milliseconds to seconds
    seconds = msec / 1000.0
    return seconds


@contextmanager
def load_vad_model():
    vad_model = VAD(sample_rate=16000)
    yield vad_model
    vad_model.empty_cache()  # Clear GPU memory

@contextmanager
def load_ctc_model():
    ctc_model = CTC(device, language='fr')
    yield ctc_model
    ctc_model.empty_cache()  # Clear GPU memory

@contextmanager
def init_asr():
    asr_class = ASR(device='cuda', compute_type='float16', asr_model="large-v3", backend="faster-whisper", beam_size=5)
    yield asr_class

@contextmanager
def load_diarization_model():
    diarization_model = diarization(device)
    yield diarization_model
    diarization_model.empty_cache()  # Clear GPU memory

import configparser

config = configparser.ConfigParser()
script_path = os.path.abspath(os.path.dirname(__file__))
config.read(os.path.join(script_path, 'config.ini'))

#files = sorted(glob.glob('/Users/research-team/theo_db/data/new_data/test/pilote.wav'))
#files = ['/root/theo_db/data/new_data/audio/28700e3c-0784-4816-9bb0-8a2d3d402ccb__01jbbm1j320wbcqsc34y8qc35d__b07840b2-884f-410d-86eb-e65501f15221.mp3']
#theo_db/data/new_data/trs_v1/trs_per_segments/28700e3c-0784-4816-9bb0-8a2d3d402ccb__01jbbm1j320wbcqsc34y8qc35d__b07840b2-884f-410d-86eb-e65501f15221.csv

files = sorted(glob.glob('/root/theo_db/data/new_data/audio/*'))
#FILTER_by_files_in = '/root/theo_db/data/new_data/json_segs_LabelStudio_option4/*.json'

#files_excluded = sorted(glob.glob(FILTER_by_files_in))
#files_excluded = [os.path.splitext(os.path.basename(files))[0] for files in files_excluded]

#files = [f for f in files if os.path.splitext(os.path.basename(f))[0] not in files_excluded]

with open('doublons.json', 'r') as fp:
    conv_duplicates = json.load(fp)

FILTER_by_duplicates_files = []
for k, v in conv_duplicates.items():
    FILTER_by_duplicates_files.extend(v[1:])

FILTER_by_duplicates_files = [os.path.splitext(os.path.basename(f))[0] for f in FILTER_by_duplicates_files]

files = [f for f in files if os.path.splitext(os.path.basename(f))[0] not in FILTER_by_duplicates_files]

files = files[:1]
#Prevent < 0.2s 

#print(os.path.join(config['Paths']['DATA'], '*.mp3'))
# Pre-filter files to process up to 400

#files = files[:1]
#print(files)

import time

start_time = time.time()

option = 4
# Step 1: create VAD references
for stereo_path in tqdm(files, desc="Processing files"):  # Wrap the loop with tqdm

    stereo_path = preprocess_audio(stereo_path, saved_dir='audio_wav')

    
    waveform, sample_rate = torchaudio.load(stereo_path)

    length_audio = int(waveform.size(1))/sample_rate

    print('Length audio', length_audio)

    # Check if stereo
    if waveform.ndim == 2 and waveform.size(0) == 2:
        client_audio_path, agent_audio_path = seperate_channels(stereo_path)

        # Load audio file
        client_waveform, sample_rate = torchaudio.load(client_audio_path)
        agent_waveform, sample_rate = torchaudio.load(agent_audio_path)

        if option==1: #whisper only

            try:
                # Step 1.2: Transcribe from VAD
                with init_asr() as ASR_class:
                    trs_client = ASR_class.process(client_audio_path)
                    trs_agent = ASR_class.process(agent_audio_path)

                    df_client = pd.DataFrame(trs_client)
                    df_agent = pd.DataFrame(trs_agent)
                    df_client['speaker']='client'
                    df_agent['speaker']='agent'

                    df = pd.concat([df_client, df_agent])
                    #sort by start time and duration (end time-start time)
                    df['duration'] = df['end'] - df['start']
                    df = df.sort_values(by=['start', 'duration'])

                    df['text'] = df['text'].fillna('')
                    
                    df.to_csv(f"/Users/research-team/theo_db/data/new_data/whisper_ts/{os.path.basename(stereo_path).replace('.wav', '')}.csv", index=False)
            except:
                print(f"Error with file {stereo_path}")    

        elif option==2: #VAD + whisper on only speech + CTC
            try:
                with load_vad_model() as vad_model:
                    client_timestamps, only_speech_client = vad_model.process(client_waveform, silence_duration=0)
                    agent_timestamps, only_speech_agent = vad_model.process(agent_waveform, silence_duration=0)

                    only_speech_path_client = vad_model.save_audio(stereo_path, only_speech_client, saved_dir='speech_only_client')
                    only_speech_path_agent = vad_model.save_audio(stereo_path, only_speech_agent, saved_dir='speech_only_agent')

                oracle_rttm_path = vad_model.save_rttm(stereo_path, client_timestamps, agent_timestamps, saved_dir='oracle_rttm_option2')

                # Step 1.2: Transcribe from VAD
                with init_asr() as ASR_class:
                    trs_client = ASR_class.process(only_speech_path_client)
                    trs_agent = ASR_class.process(only_speech_path_agent)
            
                with load_ctc_model() as CTC_model:
                    alignments_client = CTC_model.process(client_waveform, ' '.join(pd.DataFrame(trs_client)['text']))
                    alignments_agent = CTC_model.process(agent_waveform, ' '.join(pd.DataFrame(trs_agent)['text']))

                    words_csv_path_client = CTC_model.save(stereo_path, alignments_client, saved_dir='auto_trs_aligned_client')
                    words_csv_path_agent = CTC_model.save(stereo_path, alignments_agent, saved_dir='auto_trs_aligned_agent')

                    #load csv file words_csv_path_client
                    csv_client = pd.read_csv(words_csv_path_client)
                    csv_agent = pd.read_csv(words_csv_path_agent)

                    csv_client['speaker'] = 'client'
                    csv_agent['speaker'] = 'agent'
                    
                    #combine the two csv files by start time
                    csv_combined = pd.concat([csv_client, csv_agent], ignore_index=True)
                    #sort the combined csv file by start time and duration
                    csv_combined = csv_combined.sort_values(by=['start'])

                    csv_combined['text'] = csv_combined['text'].fillna('')

                    words_speakers_csv_path_oracle = align_words_with_speakers(csv_combined, oracle_rttm_path, stereo_path, saved_dir='auto_trs_aligned_speakers_oracle')
            except:
                print(f"Error with file {stereo_path}")    
        elif option==3: #VAD + whisper on only speech + CTC
            
            try:
                # Step 1.2: Transcribe from VAD
                with init_asr() as ASR_class:
                    trs_client = ASR_class.process(client_audio_path)
                    trs_agent = ASR_class.process(agent_audio_path)

                #trs_client
                segments_client = []
                for segment in trs_client:
                    trs = {}
                    
                    start = segment['start'] # to_seconds(segment.t0)
                    end = segment['end'] #to_seconds(segment.t1)
                    text = segment['text']

                    trs['start'] = start
                    trs['end'] = end
                    trs['text'] = text

                    segments_client.append(trs)
                
                #trs_client
                segments_agent = []
                for segment in trs_agent:
                    trs = {}
                    
                    start = segment['start'] # to_seconds(segment.t0)
                    end = segment['end'] #to_seconds(segment.t1)
                    text = segment['text']

                    trs['start'] = start
                    trs['end'] = end
                    trs['text'] = text

                    segments_agent.append(trs)

                from CTC_v2 import load_align_model, align
                from CTC_v2 import AlignedTranscriptionResult

                interpolate_method = "nearest" #["nearest", "linear", "ignore"]
                return_char_alignments = False
                print_progress = False

                # Load the alignment model and metadata for the specified language
                align_model, align_metadata = load_align_model('fr', device, model_name=None)

                # Perform alignment
                print(">> Performing alignment...")
                aligned_result_client: AlignedTranscriptionResult = align(
                    transcript=segments_client,  # Input transcription segments
                    model=align_model,  
                    align_model_metadata=align_metadata,  # Alignment model metadata
                    audio=client_audio_path,  # Path to the corresponding audio file
                    device=device,  
                    interpolate_method=interpolate_method,  # Interpolation method for gaps
                    return_char_alignments=return_char_alignments,  # Include character-level alignments if needed
                    print_progress=print_progress,  
                )

                print(">> Performing alignment...")
                aligned_result_agent: AlignedTranscriptionResult = align(
                    transcript=segments_agent,  # Input transcription segments
                    model=align_model,  
                    align_model_metadata=align_metadata,  # Alignment model metadata
                    audio=agent_audio_path,  # Path to the corresponding audio file
                    device=device,
                    interpolate_method=interpolate_method,  # Interpolation method for gaps
                    return_char_alignments=return_char_alignments,  # Include character-level alignments if needed
                    print_progress=print_progress,
                )

                # Clean up resources
                del align_model
                gc.collect()
                torch.cuda.empty_cache()

                # Step 1: Merge and sort segments by start time
                merged_segments = []
                for segment in aligned_result_agent['segments']:
                    merged_segments.append({'start': segment['start'], 'end': segment['end'], 'text': segment['text'], 'speaker': 'agent'})
                for segment in aligned_result_client['segments']:
                    merged_segments.append({'start': segment['start'], 'end': segment['end'], 'text': segment['text'], 'speaker': 'client'})

                # Sort by start time
                merged_segments.sort(key=lambda x: x['start'])

                # Create a DataFrame for segments
                segments_df = pd.DataFrame(merged_segments)
                #Small postprocessing
                list_of_hallucinations = [
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
                segments_df['text'] = segments_df['text'].replace(list_of_hallucinations, '', regex=True)
                segments_df['text'] = segments_df['text'].fillna('')

                # Step 2: Merge and sort words by start time
                merged_words = []
                for segment in aligned_result_agent['segments']:
                    for word in segment['words']:
                        merged_words.append({'start': word['start'], 'end': word['end'], 'word': word['word'], 'speaker': 'agent'})
                for segment in aligned_result_client['segments']:
                    for word in segment['words']:
                        merged_words.append({'start': word['start'], 'end': word['end'], 'word': word['word'], 'speaker': 'client'})

                # Sort by start time
                merged_words.sort(key=lambda x: x['start'])

                # Create a DataFrame for words
                words_df = pd.DataFrame(merged_words)

                corpus_path = os.path.dirname(os.path.dirname(stereo_path))
                output_dir = os.path.join(corpus_path, 'trs_per_segments')
                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)
                file_name = os.path.splitext(os.path.basename(stereo_path))[0] + '.csv'
                segment_df_path = os.path.join(output_dir, file_name)

                # Export to CSV
                segments_df.to_csv(segment_df_path, index=False)
                
                output_dir = os.path.join(corpus_path, 'trs_per_words')
                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)
                file_name = os.path.splitext(os.path.basename(stereo_path))[0] + '.csv'
                word_df_path = os.path.join(output_dir, file_name)


                words_df['text'] = words_df['text'].fillna('')

                words_df.to_csv(word_df_path, index=False)

                save_to_json_for_humanlabel(segment_df_path, stereo_path, length_audio, bucket_name="theo-deschamps-test", saved_dir="json_segs_LabelStudio")
            except:
                print(f"Error with file {stereo_path}")    

        elif option==4: #VAD + whisper on segments (with context)
            
            try:
                with load_vad_model() as vad_model:
                    client_timestamps, only_speech_client = vad_model.process(client_waveform, silence_duration=0)
                    agent_timestamps, only_speech_agent = vad_model.process(agent_waveform, silence_duration=0)

                    only_speech_path_client = vad_model.save_audio(stereo_path, only_speech_client, saved_dir='speech_only_client')
                    only_speech_path_agent = vad_model.save_audio(stereo_path, only_speech_agent, saved_dir='speech_only_agent')

                    #oracle_rttm_path = vad_model.save_rttm(stereo_path, client_timestamps, agent_timestamps, saved_dir='oracle_rttm')
                    results = []

                # Flag to toggle parallel processing
                use_parallel = True  # Set to False for sequential processing
                use_context = False
                # Function to limit the rolling context by character length
                def limit_rolling_context(rolling_context, max_chars):
                    """
                    Truncate the rolling context to a specified number of characters.
                    
                    Args:
                        rolling_context (str): The current rolling context.
                        max_chars (int): Maximum allowed characters for the rolling context.
                    
                    Returns:
                        str: Truncated rolling context.
                    """
                    if len(rolling_context) > max_chars:
                        # Keep only the last `max_chars` characters
                        rolling_context = rolling_context[-max_chars:]
                    return rolling_context

                def transcribe_segment(asr_instance, waveform, seg, role, rolling_context):
                    start_idx = int(seg["start"] * 16000)
                    end_idx = int(seg["end"] * 16000)
                    audio_segment = waveform[:, start_idx:end_idx]
                    
                    if use_context:
                        # Limit the rolling context by character length
                        rolling_context = limit_rolling_context(rolling_context, 200)
                    
                    # Process the audio segment with the rolling context
                    segment_trs = asr_instance.process(audio_segment, rolling_context=rolling_context, vad_filter=False)
                    
                    # Combine text from segment
                    final_trs = ' '.join([s["text"] for s in segment_trs])
                    
                    if use_context:
                        # Update rolling context by appending the transcribed text
                        rolling_context += " " + final_trs
                    
                    return {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": final_trs,
                        "speaker": role
                    }, rolling_context

                
                # Function to process segments (with or without parallelization)
                def process_segments(waveform, timestamps, role, use_parallel):
                    processed_results = []
                    rolling_context = ""  # Initialize rolling context

                    if use_parallel:
                        # Parallel processing (without rolling context)
                        with init_asr() as ASR_class, ThreadPoolExecutor() as executor:
                            future_to_segment = {
                                executor.submit(transcribe_segment, ASR_class, waveform, seg, role, rolling_context): seg
                                for seg in timestamps
                            }
                            for future in future_to_segment:
                                result, _ = future.result()
                                processed_results.append(result)
                    else:
                        # Sequential processing with rolling context
                        with init_asr() as ASR_class:
                            for seg in timestamps:
                                result, rolling_context = transcribe_segment(ASR_class, waveform, seg, role, rolling_context)
                                processed_results.append(result)

                    return processed_results

                # Process client and agent segments based on the flag
                client_results = process_segments(client_waveform, client_timestamps, role="client", use_parallel=use_parallel)
                agent_results = process_segments(agent_waveform, agent_timestamps, role="agent", use_parallel=use_parallel)

                # Combine client and agent results
                all_segments = client_results + agent_results

                # Sort the combined list by start time
                all_segments_sorted = sorted(all_segments, key=lambda seg: seg["start"])

                # Convert to DataFrame
                segments_df = pd.DataFrame(all_segments_sorted)
                
                #Small postprocessing
                list_of_hallucinations = [
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
                segments_df['text'] = segments_df['text'].replace(list_of_hallucinations, '', regex=True)

                segments_df['text'] = segments_df['text'].fillna('') #ne sert à rien par contre pd.read_csv('aa.csv', keep_default_na=False)

                corpus_path = os.path.dirname(os.path.dirname(stereo_path))
                output_dir = os.path.join(corpus_path, 'trs_per_segments_option4')
            except:
                print(f"Error with file {stereo_path}")
        
        
end_time = time.time()
execution_time = end_time - start_time
print(f"Script executed in {execution_time:.2f} seconds.")