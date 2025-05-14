# SpeechCURA

SpeechCURA is a toolkit for speech data augmentation and processing. It provides various techniques for augmenting audio data, which can be beneficial for training robust machine learning models in speech recognition tasks.

## Features

- Frequency Masking
- Time Masking
- Time Shifting
- Spectrogram Visualization

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
```

### Example

Here's a simple example of how to use the `SpecAugment` class:

```python
augmenter = SpecAugment(sample_rate=16000)
augmented_file = augmenter.frequency_masking("input.wav", "output_dir", "augmented_")
```

### Visualizing Spectrograms

You can visualize the spectrogram of an audio file using:

```python
visualize_spectrogram("path/to/audio.wav")
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.
