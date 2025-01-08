import subprocess
import os
import configparser

class ASR:
    def __init__(self, asr_model="large-v3"):
        config = configparser.ConfigParser()
        script_path = os.path.dirname(__file__)
        config.read(os.path.join(script_path, "config.ini"))
        models_path = config["Paths"]["MODELS"]
        self.whisper_cpp_path = config["Paths"]["WHISPER_CPP"]
        self.model = os.path.join(models_path, f"ggml-{asr_model}.bin")


    def process(self, file_path):

        #-nt without timestamps
        full_command = f"{self.whisper_cpp_path} -m {self.model} -f {file_path} -nt --language fr"
        query = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = query.communicate()
        #print(output)
        #print(error)
        # Process and return the output string
        decoded_str = output.decode('utf-8').strip()
        processed_str = decoded_str.replace('[BLANK_AUDIO]', '').strip()

        return processed_str

    def save(self, file_path, processed_str, saved_dir='trs'):
        
        corpus_path = os.path.dirname(os.path.dirname(file_path))
        output_dir = os.path.join(corpus_path, saved_dir)
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Generate output file path
        file_name = os.path.splitext(os.path.basename(file_path))[0] + '.txt'
        output_file = os.path.join(output_dir, file_name)

        try:
            # Write the processed string to the output file
            with open(output_file, 'w') as f:
                f.write(processed_str)
        except IOError as e:
            print(f"Error saving transcription: {e}")

if __name__ =='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='data/pissedconsumer/wav_16bit/CA0c52bd4e480925bdf3f928c85bfa1f8b.wav')
    parser.add_argument('--asr_model', type=str, default='large-v3')
    args = parser.parse_args()

    processed_str = ASR.process(args.file_path, args.asr_model)
    print(processed_str)