import os
import argparse
import subprocess

def seperate_channels(wav_file, saved_dir='audio_chan_sep'):
    """
    Preprocess an input audio file by converting it to 16 kHz WAV format.
    Handles both mono and stereo audio:
    - For stereo: splits into two files (agent and client).
    - For mono: duplicates the same file for agent and client.

    Args:
        wav_file (str): Path to the input audio file.

    Returns:
        tuple: Paths to the agent and client audio files.
    """
    corpus_path = os.path.dirname(os.path.dirname(wav_file))
    output_dir = os.path.join(corpus_path, saved_dir)
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate base file name
    base_name = os.path.splitext(os.path.basename(wav_file))[0]
    client_file = os.path.join(output_dir, f"{base_name}_client.wav")
    agent_file = os.path.join(output_dir, f"{base_name}_agent.wav")

    # Check if the input is mono or stereo
    probe_command = f"ffprobe -i {wav_file} -show_streams -select_streams a -hide_banner -loglevel error"
    probe_result = subprocess.run(probe_command, shell=True, capture_output=True, text=True)
    
    if "channels=1" in probe_result.stdout:
        # Mono file: Duplicate output
        command = f"ffmpeg -i {wav_file} -ar 16000 {client_file} -y -hide_banner -loglevel error"
        os.system(command)
        os.system(f"cp {client_file} {agent_file}")
    
    elif "channels=2" in probe_result.stdout:
        # Stereo file: Use channelsplit
        command = (
            f"ffmpeg -i {wav_file} -filter_complex "
            f"\"[0:a]channelsplit=channel_layout=stereo[left][right]; "
            f"[left]aresample=16000[{base_name}_left]; "
            f"[right]aresample=16000[{base_name}_right]\" "
            f"-map \"[{base_name}_left]\" {client_file} "
            f"-map \"[{base_name}_right]\" {agent_file} -y "
            f"-hide_banner -loglevel error"
        )
        os.system(command)
    else:
        raise ValueError("Input audio file must be either mono or stereo.")

    return client_file, agent_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_file', type=str, required=True, help="Path to the input audio file")
    args = parser.parse_args()

    try:
        client_output, agent_output = seperate_channels(args.wav_file)
        print(f"Agent audio saved at: {agent_output}")
        print(f"Client audio saved at: {client_output}")
    except Exception as e:
        print(f"Error: {e}")
