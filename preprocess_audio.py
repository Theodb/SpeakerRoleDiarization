import os
import subprocess

def preprocess_audio(file_path, saved_dir='audio_mono'):
    """
    Input: any audio type
    Output: a 16 kHz WAV file with lossless (Linear PCM) encoding
    """

    corpus_path = os.path.dirname(os.path.dirname(file_path))
    output_dir = os.path.join(corpus_path, saved_dir)
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate output file path
    output_file = os.path.join(
        output_dir,
        os.path.basename(file_path).replace('.mp3', '.wav')
    )

    # Construct and execute the ffmpeg command
    os.system(f'ffmpeg -i {file_path} -ac 1 -ar 16000 {output_file} -hide_banner -loglevel error -y')

    #'-ac', '1',      # Set audio channels to mono

    #'-c:a', 'pcm_s16le',  # Set audio codec to Linear PCM (lossless)
    #process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Check for errors
    #if process.returncode != 0:
    #    raise RuntimeError(f"Error in ffmpeg processing: {process.stderr.decode('utf-8')}")

    return output_file

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str)
    args = parser.parse_args()
    preprocess_audio(args.file_path)
