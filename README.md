# SpeechCURA

SpeechCURA is a toolkit for speech data augmentation and processing. It provides various techniques for augmenting audio data, which can be beneficial for training robust machine learning models in speech recognition tasks.

## Features

- Frequency Masking
- Time Masking
- Time Shifting
- Spectrogram Visualization
- Contrastive Augmentation (Word Masking in Audio & Text)

## Installation

To install the required packages, you can use pip:

```bash
pip install -r requirements.txt
```

## Usage

### Importing the Library

You can import the necessary classes and functions as follows:

```python
from speechcura.non_generative import SpecAugment
from speechcura.visualization import visualize_spectrogram
from speechcura.non_generative import ContrastiveAugmentation
```

### Example

Here's a simple example of how to use the `SpecAugment` class:

```python
augmenter = SpecAugment(sample_rate=16000)
augmented_file = augmenter.frequency_masking("input.wav", "output_dir", "augmented_")
```

### Contrastive Augmentation Example

The `ContrastiveAugmentation` class allows you to generate augmented audio samples by masking words in both the audio and its transcription.

```python
from speechcura.non_generative import ContrastiveAugmentation
from pathlib import Path
import json # For creating dummy transcription files in example

# Initialize the augmenter, optionally with a target sample rate
# If target_sample_rate is provided, audio will be resampled. Default is 16000 Hz.
contrast_augmenter = ContrastiveAugmentation(target_sample_rate=16000)

# --- Example 1: Augmenting a single file ---
# Assume you have 'sample_audio.wav' and its transcription 'sample_audio.json'
# Create dummy audio and transcription for demonstration:
# (In a real scenario, you would have your actual audio and transcription files)
dummy_audio_path = Path("sample_audio.wav")
dummy_transcription_path = Path("sample_audio.json")
output_single_dir = "augmented_single_output"

# Create a dummy wav file (e.g. using pydub, not shown here for brevity)
# For the example to run, create a simple wav file named sample_audio.wav
# from pydub import AudioSegment
# from pydub.generators import Sine
# Sine(440).to_audio_segment(duration=3000).export(dummy_audio_path, format="wav")


dummy_transcription_data = [
    {"text": "This", "start": 0.1, "end": 0.5},
    {"text": "is", "start": 0.6, "end": 0.8},
    {"text": "a", "start": 0.9, "end": 1.0},
    {"text": "sample", "start": 1.1, "end": 1.7},
    {"text": "sentence", "start": 1.8, "end": 2.5}
]
with open(dummy_transcription_path, 'w') as f:
    json.dump(dummy_transcription_data, f)

if dummy_audio_path.exists(): # Check if dummy audio was created
    print(f"Augmenting single file: {dummy_audio_path}")
    single_file_results = contrast_augmenter.generate_negative_samples(
        audio_path=str(dummy_audio_path),
        transcription_timestamps=dummy_transcription_data,
        n_aug=1,       # Number of augmentation rounds
        n_neg=2,       # Number of negative samples per round
        p=0.3,         # Percentage of words to mask (30%)
        output_dir=output_single_dir
    )
    print(f"Single file augmentation results (filename: masked_transcription): {single_file_results}")
    # Expected output: A dictionary like {'sample_audio_aug0_neg0.wav': 'This is [MASK] sample [MASK]', ...}
else:
    print(f"Skipping single file augmentation example: {dummy_audio_path} not found.")


# --- Example 2: Augmenting all files in a directory ---
# Assume:
# - 'input_audio_directory/' contains your .wav files
# - 'input_transcriptions_directory/' contains corresponding .json transcription files
# - 'main_augmented_output/' is where augmented files will be saved

# Create dummy directories and files for demonstration
Path("input_audio_directory").mkdir(exist_ok=True)
Path("input_transcriptions_directory").mkdir(exist_ok=True)
Path("main_augmented_output").mkdir(exist_ok=True)

# (Create some dummy .wav and .json files in these directories for the example to run)
# For example, copy sample_audio.wav and sample_audio.json into their respective input directories.
# if dummy_audio_path.exists():
#     import shutil
#     shutil.copy(dummy_audio_path, Path("input_audio_directory") / dummy_audio_path.name)
#     shutil.copy(dummy_transcription_path, Path("input_transcriptions_directory") / dummy_transcription_path.name)


print("\nAugmenting directory...")
directory_results = contrast_augmenter.augment_directory(
    input_audio_dir="input_audio_directory",
    input_transcriptions_dir="input_transcriptions_directory",
    n_aug=1,
    n_neg=1,
    p=0.25, # Mask 25% of words
    main_output_dir="main_augmented_output"
    # audio_extensions=['.wav'], # Optional: specify audio extensions
    # transcription_extension=".json" # Optional: specify transcription extension
)
print(f"Directory augmentation results (filename: masked_transcription): {directory_results}")
# Expected output: A dictionary where keys are filenames (e.g., 'sample_audio_aug0_neg0.wav')
# and values are their masked transcriptions. Files are saved in 'main_augmented_output'.

# Clean up dummy files/dirs for the example
if dummy_audio_path.exists(): dummy_audio_path.unlink()
if dummy_transcription_path.exists(): dummy_transcription_path.unlink()
# For more robust cleanup, you might use shutil.rmtree for directories if they were created solely for this example.

### Visualizing Spectrograms

You can visualize the spectrogram of an audio file using:

```python
visualize_spectrogram("path/to/audio.wav")
```
