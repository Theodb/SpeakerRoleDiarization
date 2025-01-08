import torch
import torchaudio
import torchaudio.functional as F
from torchaudio.pipelines import MMS_FA
import re
from langdetect import detect
from num2words import num2words
import argparse
import csv
import os
import configparser

import uroman as ur


class CTC:
    def __init__(self, device):

        # Instantiate the forced alignment model
        self.bundle = MMS_FA
        self.model = self.bundle.get_model(with_star=False).to(device)
        self.device = device

        config = configparser.ConfigParser()
        script_path = os.path.dirname(__file__)
        config.read(os.path.join(script_path, "config.ini"))

        self.sample_rate = int(config['PARAMS']['sr'])

        self.uroman = ur.Uroman() 
    
    # Convert numerics to words based on detected language
    def _convert_numerics(self, text, lang):
        def replace_numeric(match):
            number = match.group()
            return num2words(number, lang=lang)
        return re.sub(r'\d+', replace_numeric, text)

    def _align(self, emission, tokens):
        targets = torch.tensor([tokens], dtype=torch.int32, device=self.device)

        #print('emission', emission.shape)
        #print('targets', targets.shape)

        alignments, scores = F.forced_align(emission, targets, blank=0)

        alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
        scores = scores.exp()  # convert back to probability
        return alignments, scores

    def _unflatten(self, list_, lengths):
        assert len(list_) == sum(lengths)
        i = 0
        ret = []
        for l in lengths:
            ret.append(list_[i : i + l])
            i += l
        return ret

    # Compute average score weighted by the span length
    def _score(self, spans):
        return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)

    def _extract_word_timestamps(self, waveform, spans, num_frames, transcript):
        """
        Extract timestamps for each word in the transcript along with their audio segments.

        Args:
            waveform (Tensor): The audio waveform.
            spans (list of TokenSpan): The spans for each word.
            num_frames (int): The number of frames in the waveform.
            transcript (list of str): List of words in the transcript.
            sample_rate (int): The sample rate of the audio.

        Returns:
            list of dict: A list containing word, start time, end time, and the waveform segment for each word.
        """
        result = []
        ratio = waveform.size(1) / num_frames  # Ratio to map frames to samples

        for word, span in zip(transcript, spans):
            start_frame = span[0].start  # Extract start frame index
            end_frame = span[-1].end    # Extract end frame index
            x0 = int(ratio * start_frame)  # Start sample
            x1 = int(ratio * end_frame)    # End sample
            start_time = x0 / self.sample_rate  # Convert sample to seconds
            end_time = x1 / self.sample_rate    # Convert sample to seconds
            segment = waveform[:, x0:x1]  # Extract waveform segment

            result.append({
                "word": word,
                "start_time": start_time,
                "end_time": end_time,
                "audio_segment": segment.numpy(),  # Convert to NumPy for compatibility
            })

        return result
    
    def process(self, waveform, transcription, language='fre'):
        """
        Perform forced alignment on the given audio file and transcription.

        Args:
            waveform (str): audio file (WAV format).
            transcription (str): Transcription text.

        Returns:
            list of dict: Word-level timestamps and alignment scores.
        """

        # Detect language of the transcription
        language = detect(transcription)
        print(f"Detected language: {language}")

        # Convert numerics in the transcription
        transcription = self._convert_numerics(transcription, language)

        # Preserve original punctuation
        punctuation_indices = [(m.start(), m.group()) for m in re.finditer(r'[^\w\s]', transcription)]
        #print(f"Detected punctuation: {punctuation_indices}")
        print("punctuation not yet implemented")

        # Remove punctuation for alignment purposes
        transcription_clean = re.sub(r'[^\w\s]', '', transcription)

        # Convert transcription to Uroman
        transcription_clean = self.uroman.romanize_string(transcription_clean, language)

        # Convert transcription string into a list of words
        TRANSCRIPT = transcription_clean.lower().split()

        with torch.inference_mode():
            emission, _ = self.model(waveform.to(self.device))

        LABELS = self.bundle.get_labels(star=None)
        DICTIONARY = self.bundle.get_dict(star=None)

        tokenized_transcript = [DICTIONARY[c] for word in TRANSCRIPT for c in word]

        aligned_tokens, alignment_scores = self._align(emission, tokenized_transcript)
        token_spans = F.merge_tokens(aligned_tokens, alignment_scores)

        word_spans = self._unflatten(token_spans, [len(word) for word in TRANSCRIPT])

        num_frames = emission.size(1)

        alignments = self._extract_word_timestamps(waveform, word_spans, num_frames, TRANSCRIPT)

        return alignments

    def save(self, path_file, alignments, saved_dir='auto_trs_aligned'):
        """
        Save word alignments to a file for future diarization steps.

        Args:
            alignments (list of dict): List of word alignments with 'word', 'start_time', and 'end_time'.
        """
        corpus_path = os.path.dirname(os.path.dirname(path_file))
        output_dir = os.path.join(corpus_path, saved_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{os.path.basename(path_file).replace('.wav', '')}.csv")

        with open(output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=',')
            # Write the header
            writer.writerow(["word", "start_time", "end_time"])
            # Write each alignment
            for alignment in alignments:
                writer.writerow([alignment["word"], alignment["start_time"], alignment["end_time"]])

        return output_file
    
    def empty_cache(self):
        del self.model
        torch.cuda.empty_cache()

    
    #word_alignments = compute_alignments(emissions, transcript_list, dictionary, self.sample_rate)
#
    ## Reinsert punctuation into the aligned transcription
    #aligned_transcription = ' '.join([wa['word'] for wa in word_alignments])
    #for idx, p in punctuation_indices:
    #    aligned_transcription = (
    #        aligned_transcription[:idx] + p + aligned_transcription[idx:]
    #    )
#
    #print(f"Aligned Transcription with Punctuation: {aligned_transcription}")
#
    #return word_alignments

if __name__ == "__main__":
    pass
    #parser = argparse.ArgumentParser(description="Perform forced alignment of words with audio.")
    #parser.add_argument("--path_file", type=str, help="Path to the WAV file.", default='/Users/theodeschampsberger/travail/data/pissed_consumer/wav/CA3bb7ab0c522ad80d6adb16ad40d82041.wav')
    #parser.add_argument("--transcription", type=str, default="Thanks for calling Wayfair. This call may be recorded to help our team serve you better. To get help with an order you've already placed, press 2 now. To place an order or to get help with products you're interested in, we're sorry, we're having some trouble pulling up your information. We'll need to ask you a few questions to connect you with the right team member. Is the phone number listed on your order different from the phone number you're calling from? If so, press 1. If not, press 2. A member of our team will be with you shortly. Thank you for calling Wayfair. My name is Jessica. How can I help you today? I'm calling about a previous order that I just placed that I got in the mail today and y'all sent out the wrong item. Y'all didn't send me what I paid for. I'm sorry to hear that. I'm happy to assist you. Can I have your name, please? My name is Chevelle Dukes. Thank you, Chevelle. Just a moment here. My order came through with your number. Do you have the order number or the email address so I can access your order? My order number is 428-253-3642. Thank you, Chevelle. One moment. Can I get you to verify the billing address? As well as the email address that we have here on file. 744 Virginia Avenue, Akron, Ohio, 44306. Chevelle1989.sd at gmail.com. Thank you. This is for the 5x7 rug? Yes. Okay. I'm going to go ahead and send you... Are you able to receive text? Text messages or email, which you prefer? Yeah, I can receive text messages. Okay. Yeah, y'all sent a whole... Yeah, y'all sent a whole different color rug. That ain't the rug I ordered. Okay. Again, I'm happy to help. I just need to get a photo and then some details off the packaging. I'm happy to get this report filed for you. We have a number ending in 2575. SMS message? Okay. It's 28... What'd you say? Ending in what? 2575. Yes. Yeah. Okay. I've sent the request and so let me know when the photo says that it's uploaded. And then on the package itself, there should be an item number, part number, or SKU number. It's all numbers. You can give me that number off the package. Or on the tag itself of the rug. Okay. I'm going to have to open this rug up. So you got to give me a second because I didn't... It should be on the outside of the package. Okay. You said on the outside of the packaging? Yeah. Sometimes, yeah, it's on the packaging on the label. But it's okay. I can wait while you open it. Not a problem. Okay. Not a problem. You said you want the SKU number? That's what you said? Yeah. From the item. It's a part item or SKU number. It's listed on either the label on the package or the label on the rug. Okay. Okay. It looks like it's all numerical. That's correct. One second. You're fine. I'm here. Okay. Okay. Okay. Okay. Okay. Okay. Okay. Okay. Okay. Okay. Okay. Okay. Okay. Okay. Okay. Okay. What color did you receive? you")
    #args = parser.parse_args('')
#
    ## Perform forced alignment
    #alignments = CTC(args.path_file, args.transcription)
#
    #for alignment in alignments:
    #    print(f"{alignment['word']}\t{alignment['start_time']}\t{alignment['end_time']}")