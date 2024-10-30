import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler
from openpyxl import load_workbook

# Function to extract features from audio segment that has bird audio by using VAD
def advanced_vad(y, sr, frame_length=512, hop_length=128, flux_threshold=0.01):
    spectral_flux = np.diff(librosa.feature.spectral_flatness(y=y, n_fft=frame_length, hop_length=hop_length), axis=1)
    spectral_flux = np.pad(spectral_flux, ((0, 0), (1, 0)), mode='constant')
    flux_energy = np.sum(np.abs(spectral_flux), axis=0)

    # Determine threshold
    threshold = flux_threshold * np.max(flux_energy)
    call_indices = np.where(flux_energy > threshold)[0]

    # Create mask
    mask = np.zeros_like(flux_energy, dtype=bool)
    mask[call_indices] = True
    return mask

# Function to extract features from audio
def extract_features(y, sr):
    # Resample audio to a consistent high sampling rate
    target_sr = 22050
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Apply VAD
    speech_mask = advanced_vad(y, sr, frame_length=512, hop_length=128)
    if not np.any(speech_mask):
        print("No bird call detected")
        return None

    features = []  # To store features for each segment

    # Set parameters for feature extraction
    n_fft = 11025
    hop_length = 5512
    n_mels = 40
    f_max = sr / 2

    for i in range(0, len(y) - n_fft, hop_length):
        segment = y[i:i + n_fft]

        # Ensure we skip segments without voice activity
        if np.all(segment == 0) or not speech_mask[i // hop_length]:
            continue

        # Check segment length to avoid FFT errors
        if len(segment) < n_fft:
            print(f"Segment too short for FFT: {len(segment)}")
            continue

        try:
            # Extract features
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=segment, frame_length=n_fft, hop_length=hop_length))
            rms = np.mean(librosa.feature.rms(y=segment, frame_length=n_fft, hop_length=hop_length))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length))
            mfccs = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length, fmax=f_max), axis=1)

            # Concatenate features into a single array
            segment_features = np.concatenate((
                [zero_crossing_rate, rms, spectral_centroid],
                mfccs
            ))

            # Append segment's features to list
            features.append(segment_features)

        except Exception as e:
            print(f"Error extracting features from segment: {e}")
            continue

    # If no features were extracted, return None
    if not features:
        print("No features extracted")
        return None

    # Convert list of features to numpy array for uniform structure
    features_array = np.array(features)
    return features_array

# Function to process audio segments and extract features
def process_audio_segments(audio, sr, species, segment_file_name, output_file):
    try:
        feature_columns = ['filename', 'species'] + [f'feature_{i}' for i in range(16)]
        features = extract_features(audio, sr)

        if features is not None:
            # Each entry in `features` is a row of feature values for the DataFrame
            segments_df = pd.DataFrame(features, columns=feature_columns[2:])
            segments_df.insert(0, 'filename', segment_file_name)
            segments_df.insert(1, 'species', species)

            scaler = StandardScaler()
            segments_df.iloc[:, 2:] = scaler.fit_transform(segments_df.iloc[:, 2:])

            # Check if output file exists and load existing data if it does
            if os.path.exists(output_file):
                with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
                    existing_df = pd.read_excel(output_file)
                    combined_df = pd.concat([existing_df, segments_df], ignore_index=True)
                    combined_df.to_excel(writer, index=False)
            else:
                # If file doesn't exist, create a new one with the extracted features
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    segments_df.to_excel(writer, index=False)
            print(f"Features saved for {segment_file_name}")
        else:
            print(f"No valid features extracted for {segment_file_name}")
    except Exception as e:
        print(f"Error processing audio segment for {segment_file_name}: {e}")

# Load audio metadata from Excel and process each file's segments
def load_and_process_audio_segments(excel_file, output_dir, output_file):
    df = pd.read_excel(excel_file)

    # Check if the output file exists, and load already processed segments if so
    if os.path.exists(output_file):
        existing_df = pd.read_excel(output_file)
        if 'filename' in existing_df.columns:
            processed_segments = set(existing_df['filename'].tolist())
        else:
            processed_segments = set()
    else:
        processed_segments = set()

    for index, row in df.iterrows():
        recording_id = row['id_num']
        species = row['en']
        
        for i in range(0, 50):  # Adjust range if you know the exact segment count
            segment_file_name = f"{recording_id}_segment_{i}.mp3"
            file_path = os.path.join(output_dir, segment_file_name)
            
            # Skip if the segment has already been processed
            if segment_file_name in processed_segments:
                print(f"Skipping already processed segment: {segment_file_name}")
                continue
            
            if os.path.exists(file_path):
                print(f"Processing {file_path} for species {species}...")
                audio, sr = librosa.load(file_path, sr=None)
                process_audio_segments(audio, sr, species, segment_file_name, output_file)
            else:
                print(f"Segment file not found: {segment_file_name}")

# Main execution
try:
    excel_file_path = 'C:\\Users\\NEW\\Documents\\Python Scripts\\ML\\birds_india.xlsx'
    output_directory = 'D:\\refined_bird_calls'
    output_file = 'C:\\Users\\NEW\\Documents\\Python Scripts\\bird_species_features_segments.xlsx'
    
    load_and_process_audio_segments(excel_file_path, output_directory, output_file)
except Exception as e:
    print(f"Critical error during execution: {e}")
