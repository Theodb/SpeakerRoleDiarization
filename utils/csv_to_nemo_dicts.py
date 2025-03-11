import csv
import os

def csv_to_nemo_dicts(csv_paths):
    """
    Extract word_hyp and word_ts_hyp dictionaries from one or more CSV files.

    Args:
        csv_paths (list): List of paths to the CSV files.

    Returns:
        tuple: word_hyp (dict), word_ts_hyp (dict)
    """
    # Initialize the dictionaries
    word_hyp = {}
    word_ts_hyp = {}

    # Process each CSV path
    for csv_path in csv_paths:
        # Use the file name (without extension) as the audio ID
        audio_id = os.path.basename(csv_path).split('.')[0]

        # Open and read the CSV file
        with open(csv_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)

            # Skip the header
            next(csvreader)

            # Prepare lists to store data for this audio ID
            words = []
            timestamps = []

            # Process each row
            for row in csvreader:
                word = row[0]
                start_time = float(row[1])
                end_time = float(row[2])

                words.append(word)
                timestamps.append([start_time, end_time])

            # Add to the dictionaries
            word_hyp[audio_id] = words
            word_ts_hyp[audio_id] = timestamps

    return word_hyp, word_ts_hyp