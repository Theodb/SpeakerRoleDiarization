import subprocess
import os
import logging

def stemming(wav_file, *args):
    audio_path = os.path.dirname(os.path.dirname(wav_file))
    stemming_path = os.path.join(audio_path, "audio_stems")
    os.makedirs(stemming_path, exist_ok=True)

    return_code = os.system(f'python3 -m demucs.separate --device cpu -n htdemucs_ft --two-stems=vocals "{wav_file}" -o {stemming_path}')
    #process = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #output, error = process.communicate()
    #print(output)
    
    if return_code != 0:
        logging.warning(
            "Source splitting failed, using original audio file. "
            "Use --no-stem argument to disable it."
        )
        vocal_target = wav_file
    else:
        vocal_target = os.path.join(
            stemming_path,
            "htdemucs_ft",
            os.path.splitext(os.path.basename(wav_file))[0],
            "vocals.wav",
        )
    
    return vocal_target

if __name__ =='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_file', type=str)
    args = parser.parse_args()
    stemming(args.wav_file)