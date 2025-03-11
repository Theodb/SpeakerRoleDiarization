import pandas as pd
from utils.process_rttm import ProcessRTTM
import os

def align_words_with_speakers(words_csv_path, rttm_file_path, path_file, saved_dir='aligned_words_speakers'):
    
    corpus_path = os.path.dirname(os.path.dirname(path_file))
    output_dir = os.path.join(corpus_path, saved_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{os.path.basename(path_file).replace('.wav', '')}.csv")
    
    # Load the words CSV
    words_df = pd.read_csv(words_csv_path)

    # Load and encode the RTTM file
    process_rttm = ProcessRTTM(path=rttm_file_path, load=True)
    process_rttm.encode_rttm()

    aligned_data = []

    for _, word_row in words_df.iterrows():
        word_start = word_row["start_time"]
        word_end = word_row["end_time"]
        word = word_row["word"]

        best_overlap = 0
        best_speaker = "Unknown"  # Default value if no overlap is found

        # Check overlaps with all speaker segments
        for rttm_line in process_rttm.rttmLines:
            segment_start = rttm_line.startTime
            segment_end = rttm_line.endTime

            # Calculate overlap
            overlap_start = max(word_start, segment_start)
            overlap_end = min(word_end, segment_end)
            overlap_duration = max(0, overlap_end - overlap_start)

            # Update the best speaker if the overlap is longer
            if overlap_duration > best_overlap:
                best_overlap = overlap_duration
                best_speaker = rttm_line.speakerName

        # Add the aligned word with speaker information
        aligned_data.append({
            "word": word,
            "start_time": word_start,
            "end_time": word_end,
            "speaker": best_speaker
        })

    # Convert to DataFrame and save
    aligned_df = pd.DataFrame(aligned_data)
    aligned_df.to_csv(output_file, index=False)

    return output_file