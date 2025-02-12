# Audio Processing and Transcription Pipeline

This repository implements a modular pipeline for processing stereo audio files to produce speaker-attributed transcriptions. The pipeline includes audio preprocessing, channel separation, voice activity detection (VAD), automatic speech recognition (ASR) using Whisper, CTC-based alignment, diarization, and word-to-speaker alignment. It is designed to work on French audio content but can be adapted for other languages with minor changes.

## Overview

The main functionalities of the pipeline include:

- **Preprocessing**: Convert audio files (e.g., MP3) to WAV format.
- **Channel Separation**: Split stereo audio into two mono channels (typically representing a client and an agent).
- **Voice Activity Detection (VAD)**: Detect speech segments in each channel.
- **Transcription & Alignment Options**:
  - **Option 1 (Whisper-only)**: Transcribe each channel using Whisper.
  - **Option 2 (VAD + Whisper + CTC)**: Extract only-speech segments via VAD, then transcribe and align words using a CTC model.
  - **Option 3 (VAD + Whisper + CTC_v2)**: Use a modified alignment process after VAD and Whisper transcription.
  - **Option 4 (VAD + Whisper on segments with context)**: Transcribe VAD-derived segments (with optional rolling context) using parallel processing.
- **Output Generation**: Save transcription outputs as CSV files and prepare JSON files for integration with Label Studio (for human labeling).

The script also automatically detects available hardware (CUDA, MPS, or CPU) and sets the computation device accordingly.

## Features

- **Modular Design with Context Managers**: Models for ASR, VAD, CTC, and diarization are loaded as needed and cleaned up to free GPU memory.
- **Parallel Processing**: Use of `ThreadPoolExecutor` for parallel transcription of audio segments.
- **Multiple Processing Options**: Easily switch between different transcription/alignment strategies by setting the `option` variable.
- **Configurable**: Reads external configuration via `config.ini` and uses a JSON file (`doublons.json`) to filter duplicate files.
- **Integration Ready**: Prepares outputs that can be ingested by Label Studio for manual review and correction.