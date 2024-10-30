import os
import pandas as pd
import librosa
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from pydub import AudioSegment

# Step 1: Read the Excel file for bird species and audio paths
def load_audio_metadata(excel_file):
    try:
        df = pd.read_excel(excel_file)
        df1 = df.head(4600)  # Limit to 20 rows for now
        audio_paths = []
        for index, row in df1.iterrows():
            recording_id = row['id_num']
            file_name = os.path.join('D:\\bird_songs', f'{recording_id}.mp3')
            if os.path.exists(file_name):
                audio_paths.append(file_name)
            else:
                print(f"File not found: {file_name}")
        species = df['en'].tolist()  # Assuming 'en' column contains species names
        return audio_paths, species
    except Exception as e:
        print(f"Error loading metadata from {excel_file}: {e}")
        return [], []

# Step 2: Preprocess the audio file by applying a high-pass filter to reduce background noise
def butter_highpass_filter(data, cutoff=1000, fs=22050, order=5):
    try:
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        y = lfilter(b, a, data)
        return y
    except Exception as e:
        print(f"Error applying high-pass filter: {e}")
        return data  # Return unfiltered data if error occurs

def preprocess_audio(audio_file):
    try:
        audio, sr = librosa.load(audio_file, sr=None)
        filtered_audio = butter_highpass_filter(audio, cutoff=1000, fs=sr)
        return filtered_audio, sr
    except Exception as e:
        print(f"Error preprocessing audio file {audio_file}: {e}")
        return None, None

# Step 3: Advanced Voice Activity Detection (VAD) using energy-based segmentation
def detect_bird_calls(audio, sr, threshold=0.02):
    try:
        # Calculate the root mean square (RMS) energy of the audio
        energy = librosa.feature.rms(y=audio)[0]  # 'y' specifies the audio signal

        # Split the audio into segments based on silence and sound
        segments = librosa.effects.split(audio, top_db=25)  # Splitting non-silent segments
        
        # Filter segments based on energy threshold
        bird_segments = [segment for segment in segments if np.mean(energy[int(segment[0] // 512):int(segment[1] // 512)]) > threshold]
        
        return bird_segments
    except Exception as e:
        print(f"Error detecting bird calls in the audio: {e}")
        return []

# Step 4: Ensure segments are at least 1 to 2 seconds long by merging short segments
def segment_audio(audio, sr, segments, min_duration=1.0, max_duration=2.0):
    try:
        min_samples = int(min_duration * sr)  # Minimum segment duration in samples
        max_samples = int(max_duration * sr)  # Maximum segment duration in samples
        segmented_clips = []
        current_segment = None
        
        for start, end in segments:
            if current_segment is None:
                current_segment = [start, end]
            else:
                # Merge segments if too short
                if end - current_segment[0] < min_samples:
                    current_segment[1] = end  # Extend current segment
                else:
                    if current_segment[1] - current_segment[0] >= min_samples:
                        segmented_clips.append(audio[current_segment[0]:current_segment[1]])
                    current_segment = [start, end]

        # Append the last segment if it meets the minimum duration requirement
        if current_segment and (current_segment[1] - current_segment[0] >= min_samples):
            segmented_clips.append(audio[current_segment[0]:current_segment[1]])

        return segmented_clips
    except Exception as e:
        print(f"Error segmenting the audio: {e}")
        return []

# Step 5: Visualize and analyze the detected segments
def visualize_segments(audio, sr, segments):
    try:
        '''
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(audio, sr=sr, alpha=0.5)
        for start, end in segments:
            plt.axvspan(start / sr, end / sr, color='red', alpha=0.3)
        plt.title('Detected Bird Call Segments')
        plt.show()
        '''
        print("hi done")
    except Exception as e:
        print(f"Error visualizing the segments: {e}")

# Step 6: Save the refined audio segments
def save_segments(audio_segments, sr, output_dir, file_name):
    try:
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, segment in enumerate(audio_segments):
            # Define the output file path
            output_path = f"{output_dir}\\{file_name}_segment_{i}.mp3"

            # Check if the segment file already exists
            if os.path.exists(output_path):
                print(f"Segment {output_path} already exists. Skipping...")
                continue

            # Convert float32 audio data to int16 format (Pydub works with int16)
            segment_int16 = np.int16(segment * 32767)  # Convert float [-1, 1] to int16 range

            # Create an AudioSegment from the numpy array
            segment_audio = AudioSegment(
                segment_int16.tobytes(),  # Convert to bytes
                frame_rate=sr,            # Sampling rate
                sample_width=2,           # 2 bytes per sample (16-bit PCM)
                channels=1                # Assuming mono audio
            )

            # Export the segment as an MP3 file
            print(f"Saving segment to: {output_path}")
            segment_audio.export(output_path, format="mp3")
            
    except Exception as e:
        print(f"Error saving audio segments: {e}")

'''
def save_segments(audio_segments, sr, output_dir, file_name):
    try:
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, segment in enumerate(audio_segments):
            # Convert float32 audio data to int16 format (Pydub works with int16)
            segment_int16 = np.int16(segment * 32767)  # Convert float [-1, 1] to int16 range

            # Create an AudioSegment from the numpy array
            segment_audio = AudioSegment(
                segment_int16.tobytes(),  # Convert to bytes
                frame_rate=sr,            # Sampling rate
                sample_width=2,           # 2 bytes per sample (16-bit PCM)
                channels=1                # Assuming mono audio
            )
            # Export the segment as an MP3 file
            output_path = f"{output_dir}\\{file_name}_segment_{i}.mp3"
            print(f"Saving segment to: {output_path}")
            segment_audio.export(output_path, format="mp3")
            print(f"Saved segment {i} to {output_dir}/{file_name}_segment_{i}.mp3")
            
    except Exception as e:
        print(f"Error saving audio segments: {e}")
'''

# Main logic for processing all files
def process_bird_audio_files(excel_file, output_dir):
    audio_paths, species_list = load_audio_metadata(excel_file)

    for i, audio_file in enumerate(audio_paths):
        try:
            print(f"Processing {audio_file} for species {species_list[i]}...")
            preprocessed_audio, sr = preprocess_audio(audio_file)

            if preprocessed_audio is None:
                print(f"Skipping file {audio_file} due to preprocessing failure.")
                continue

            bird_segments = detect_bird_calls(preprocessed_audio, sr)
            segmented_clips = segment_audio(preprocessed_audio, sr, bird_segments)

            if segmented_clips:
                visualize_segments(preprocessed_audio, sr, bird_segments)
                save_segments(segmented_clips, sr, output_dir, os.path.basename(audio_file).split('.')[0])
            else:
                print(f"No bird segments detected in {audio_file}.")
        except Exception as e:
            print(f"Error processing file {audio_file}: {e}")

# Execution
try:
    excel_file_path = 'C:\\Users\\NEW\\Documents\\Python Scripts\\ML\\birds_india.xlsx'
    output_directory = 'D:\\refined_bird_calls'
    process_bird_audio_files(excel_file_path, output_directory)
except Exception as e:
    print(f"Critical error during execution: {e}")
