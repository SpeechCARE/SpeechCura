import os
import torch
import torchaudio
import random
import numpy as np


class SpecAugment:
    """
    A class for applying various spectrogram augmentation techniques to audio files.
    
    This class provides methods for frequency masking, time masking, and time shifting,
    which are common augmentation techniques used in speech recognition and audio processing.
    
    Attributes:
        sample_rate (int): Sample rate of the audio.
        seed (int): Random seed for reproducibility.
        n_fft (int): Size of FFT.
        n_stft (int): Size of STFT. If None, computed as n_fft//2 + 1.
        n_mels (int): Number of mel filterbanks.
        hop_length (int): Length of hop between STFT windows.
        win_length (int): Window size for STFT.
        device (str): Device to use for computation ('cuda' or 'cpu').
    """
    
    def __init__(self, sample_rate=16000, seed=133, n_fft=2048, n_stft=None, 
                 n_mels=128, hop_length=512, win_length=2048):
        """
        Initialize the SpecAugment class with common parameters.
        
        Args:
            sample_rate (int, optional): Sample rate of the audio. Defaults to 16000.
            seed (int, optional): Random seed for reproducibility. Defaults to 133.
            n_fft (int, optional): Size of FFT. Defaults to 2048.
            n_stft (int, optional): Size of STFT. If None, computed as n_fft//2 + 1. Defaults to None.
            n_mels (int, optional): Number of mel filterbanks. Defaults to 128.
            hop_length (int, optional): Length of hop between STFT windows. Defaults to 512.
            win_length (int, optional): Window size for STFT. Defaults to 2048.
        """
        self.sample_rate = sample_rate
        self.seed = seed
        self.n_fft = n_fft
        self.n_stft = n_fft // 2 + 1 if n_stft is None else n_stft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Set random seeds for reproducibility
        self._set_seeds(seed)
    
    def _set_seeds(self, seed):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def _load_audio(self, wav_file):
        """
        Load and preprocess an audio file.
        
        Args:
            wav_file (str): Path to the input audio file.
            
        Returns:
            torch.Tensor: Preprocessed waveform tensor.
        """
        # Load the audio file
        waveform, sr = torchaudio.load(wav_file)
        
        # If the sample rate doesn't match the expected one, resample
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate).to(self.device)
            waveform = resampler(waveform)
        
        # Ensure waveform is 2D (batch_size, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        return waveform.to(self.device)
    
    def _get_spectrogram(self, waveform):
        """
        Convert waveform to spectrogram.
        
        Args:
            waveform (torch.Tensor): Input waveform tensor.
            
        Returns:
            torch.Tensor: Spectrogram representation.
        """
        spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            f_max=self.sample_rate/2,
        ).to(self.device)
        
        return spectrogram_transform(waveform)
    
    def _spectrogram_to_waveform(self, spectrogram):
        """
        Convert spectrogram back to waveform.
        
        Args:
            spectrogram (torch.Tensor): Input spectrogram tensor.
            
        Returns:
            torch.Tensor: Reconstructed waveform.
        """
        # Convert back to linear spectrogram
        inverse_transform = torchaudio.transforms.InverseMelScale(
            n_stft=self.n_stft,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            f_max=self.sample_rate/2,
        ).to(self.device)
        linear_spec = inverse_transform(spectrogram)
        
        # Use Griffin-Lim to reconstruct the waveform
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_iter=64  # Number of iterations for Griffin-Lim
        ).to(self.device)
        
        return griffin_lim(linear_spec)
    
    def _save_audio(self, waveform, output_dir, aug_label, wav_file):
        """
        Save the augmented audio.
        
        Args:
            waveform (torch.Tensor): Augmented waveform tensor.
            output_dir (str): Directory where the augmented audio will be saved.
            aug_label (str): Label to prefix the output filename.
            wav_file (str): Original audio file path.
            
        Returns:
            str: Path to the saved augmented audio file.
        """
        os.makedirs(output_dir, exist_ok=True)
        name = aug_label + os.path.basename(wav_file)
        output_path = os.path.join(output_dir, name)
        torchaudio.save(output_path, waveform.cpu(), self.sample_rate)
        
        return output_path
    
    def frequency_masking(self, wav_file, output_dir, aug_label, mask_param=60, mask_type="zero"):
        """
        Apply frequency masking augmentation to an audio file.
        
        This method applies frequency masking to a spectrogram representation of an audio file
        and then reconstructs the audio. Frequency masking zeros out or replaces with mean values
        a continuous band of frequency bins in the spectrogram, which helps in training
        models to be robust to variations in the frequency domain.
        
        Args:
            wav_file (str): Path to the input audio file.
            output_dir (str): Directory where the augmented audio will be saved.
            aug_label (str): Label to prefix the output filename.
            mask_param (int, optional): Maximum width of the frequency mask. Defaults to 60.
            mask_type (str, optional): Type of masking to apply - "zero" for zero masking
                                      or "mean" for mean masking. Defaults to "zero".
        
        Returns:
            str: Path to the saved augmented audio file.
        """
        self._set_seeds(self.seed)
        
        # Load audio and convert to spectrogram
        waveform = self._load_audio(wav_file)
        spectrogram = self._get_spectrogram(waveform)
        
        # Apply frequency masking based on mask_type
        if mask_type.lower() == "zero":
            # Manual masking with zeros
            num_freq_bins, time_steps = spectrogram.shape[-2:]
            mask_width = torch.randint(0, mask_param, (1,)).item()  # Random mask width
            mask_start = torch.randint(0, num_freq_bins - mask_width, (1,)).item()  # Random start index
            
            # Clone spectrogram to avoid in-place modification
            masked_spectrogram = spectrogram.clone()
            masked_spectrogram[..., mask_start:mask_start + mask_width, :] = 0
        elif mask_type.lower() == "mean":
            # Use torchaudio's FrequencyMasking transform
            freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=mask_param).to(self.device)
            masked_spectrogram = freq_mask(spectrogram)
        else:
            raise ValueError("mask_type must be either 'zero' or 'mean'")
        
        # Convert back to waveform
        masked_waveform = self._spectrogram_to_waveform(masked_spectrogram)
        
        # Save the augmented audio
        return self._save_audio(masked_waveform, output_dir, aug_label, wav_file)
    
    def time_masking(self, wav_file, output_dir, aug_label, mask_param=60, mask_type="zero"):
        """
        Apply time masking augmentation to an audio file.
        
        This method applies time masking to a spectrogram representation of an audio file
        and then reconstructs the audio. Time masking zeros out or replaces with mean values
        a continuous segment of time frames in the spectrogram, which helps in training
        models to be robust to variations in the time domain.
        
        Args:
            wav_file (str): Path to the input audio file.
            output_dir (str): Directory where the augmented audio will be saved.
            aug_label (str): Label to prefix the output filename.
            mask_param (int, optional): Maximum width of the time mask. Defaults to 60.
            mask_type (str, optional): Type of masking to apply - "zero" for zero masking
                                      or "mean" for mean masking. Defaults to "zero".
        
        Returns:
            str: Path to the saved augmented audio file.
        """
        self._set_seeds(self.seed)
        
        # Load audio and convert to spectrogram
        waveform = self._load_audio(wav_file)
        spectrogram = self._get_spectrogram(waveform)
        
        # Apply time masking based on mask_type
        if mask_type.lower() == "zero":
            # Manual masking with zeros
            num_freq_bins, time_steps = spectrogram.shape[-2:]
            mask_width = torch.randint(0, mask_param, (1,)).item()  # Random mask width
            mask_start = torch.randint(0, time_steps - mask_width, (1,)).item()  # Random start index
            
            # Clone spectrogram to avoid in-place modification
            masked_spectrogram = spectrogram.clone()
            masked_spectrogram[..., :, mask_start:mask_start + mask_width] = 0
        elif mask_type.lower() == "mean":
            # Use torchaudio's TimeMasking transform
            time_mask = torchaudio.transforms.TimeMasking(time_mask_param=mask_param).to(self.device)
            masked_spectrogram = time_mask(spectrogram)
        else:
            raise ValueError("mask_type must be either 'zero' or 'mean'")
        
        # Convert back to waveform
        masked_waveform = self._spectrogram_to_waveform(masked_spectrogram)
        
        # Save the augmented audio
        return self._save_audio(masked_waveform, output_dir, aug_label, wav_file)
    
    def time_shifting(self, wav_file, output_dir, aug_label, shift_max=0.5):
        """
        Apply time shifting augmentation to an audio file.
        
        This method shifts the audio in the time domain by a random amount.
        Time shifting helps in training models to be robust to variations in
        the temporal positioning of sounds within an audio sample.
        
        Args:
            wav_file (str): Path to the input audio file.
            output_dir (str): Directory where the augmented audio will be saved.
            aug_label (str): Label to prefix the output filename.
            shift_max (float, optional): Maximum shift as a fraction of total length. Defaults to 0.5.
        
        Returns:
            str: Path to the saved augmented audio file.
        """
        self._set_seeds(self.seed)
        
        # Load the audio
        waveform = self._load_audio(wav_file)
        
        # Get the number of samples
        num_samples = waveform.shape[-1]
        
        # Calculate the maximum possible shift
        max_shift = int(num_samples * shift_max)
        
        # Generate a random shift amount
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        
        # Shift the waveform
        shifted_waveform = torch.roll(waveform, shift, dims=-1)
        
        # Save the augmented audio
        return self._save_audio(shifted_waveform, output_dir, aug_label, wav_file)

    def apply_augmentations(self, wav_file, output_dir, augmentations=None, aug_prefix="aug_"):
        """
        Apply multiple augmentations to an audio file.
        
        Args:
            wav_file (str): Path to the input audio file.
            output_dir (str): Directory where the augmented audio will be saved.
            augmentations (list, optional): List of augmentations to apply.
                                           Each item should be a dict with keys 'type' and 'params'.
                                           Defaults to None (applies all augmentations with default params).
            aug_prefix (str, optional): Prefix for augmentation labels. Defaults to "aug_".
        
        Returns:
            dict: Dictionary mapping augmentation types to output file paths.
        """
        if augmentations is None:
            # Default: apply all augmentations with default parameters
            augmentations = [
                {'type': 'freq_mask', 'params': {}},
                {'type': 'time_mask', 'params': {}},
                {'type': 'time_shift', 'params': {}}
            ]
        
        results = {}
        
        for aug in augmentations:
            aug_type = aug['type']
            params = aug.get('params', {})
            
            if aug_type == 'freq_mask':
                label = f"{aug_prefix}freq_mask_"
                results['freq_mask'] = self.frequency_masking(
                    wav_file, output_dir, label, **params
                )
            elif aug_type == 'time_mask':
                label = f"{aug_prefix}time_mask_"
                results['time_mask'] = self.time_masking(
                    wav_file, output_dir, label, **params
                )
            elif aug_type == 'time_shift':
                label = f"{aug_prefix}time_shift_"
                results['time_shift'] = self.time_shifting(
                    wav_file, output_dir, label, **params
                )
            else:
                raise ValueError(f"Unknown augmentation type: {aug_type}")
        
        return results


# Legacy functions for backward compatibility
def frequency_masking(wav_file, output_dir, aug_label, mask_param=60, seed=133, mask_type="zero", 
                      n_fft=2048, n_stft=None, n_mels=128, sample_rate=16000, hop_length=512, win_length=2048):
    """
    Apply frequency masking augmentation to an audio file.
    
    This function is maintained for backward compatibility.
    It's recommended to use the SpecAugment class for new code.
    
    Args:
        wav_file (str): Path to the input audio file
        output_dir (str): Directory where the augmented audio will be saved
        aug_label (str): Label to prefix the output filename
        mask_param (int, optional): Maximum width of the frequency mask. Defaults to 60.
        seed (int, optional): Random seed for reproducibility. Defaults to 133.
        mask_type (str, optional): Type of masking to apply - "zero" for zero masking
                                  or "mean" for mean masking. Defaults to "zero".
        n_fft (int, optional): Size of FFT. Defaults to 2048.
        n_stft (int, optional): Size of STFT. If None, computed as n_fft//2 + 1. Defaults to None.
        n_mels (int, optional): Number of mel filterbanks. Defaults to 128.
        sample_rate (int, optional): Sample rate of the audio. Defaults to 16000.
        hop_length (int, optional): Length of hop between STFT windows. Defaults to 512.
        win_length (int, optional): Window size for STFT. Defaults to 2048.
    
    Returns:
        str: Path to the saved augmented audio file
    """
    augmenter = SpecAugment(
        sample_rate=sample_rate, 
        seed=seed,
        n_fft=n_fft,
        n_stft=n_stft,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length
    )
    return augmenter.frequency_masking(wav_file, output_dir, aug_label, mask_param, mask_type)

def time_masking(wav_file, output_dir, aug_label, mask_param=60, seed=133, mask_type="zero",
                n_fft=2048, n_stft=None, n_mels=128, sample_rate=16000, hop_length=512, win_length=2048):
    """
    Apply time masking augmentation to an audio file.
    
    This function is maintained for backward compatibility.
    It's recommended to use the SpecAugment class for new code.
    
    Args:
        wav_file (str): Path to the input audio file
        output_dir (str): Directory where the augmented audio will be saved
        aug_label (str): Label to prefix the output filename
        mask_param (int, optional): Maximum width of the time mask. Defaults to 60.
        seed (int, optional): Random seed for reproducibility. Defaults to 133.
        mask_type (str, optional): Type of masking to apply - "zero" for zero masking
                                  or "mean" for mean masking. Defaults to "zero".
        n_fft (int, optional): Size of FFT. Defaults to 2048.
        n_stft (int, optional): Size of STFT. If None, computed as n_fft//2 + 1. Defaults to None.
        n_mels (int, optional): Number of mel filterbanks. Defaults to 128.
        sample_rate (int, optional): Sample rate of the audio. Defaults to 16000.
        hop_length (int, optional): Length of hop between STFT windows. Defaults to 512.
        win_length (int, optional): Window size for STFT. Defaults to 2048.
    
    Returns:
        str: Path to the saved augmented audio file
    """
    augmenter = SpecAugment(
        sample_rate=sample_rate, 
        seed=seed,
        n_fft=n_fft,
        n_stft=n_stft,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length
    )
    return augmenter.time_masking(wav_file, output_dir, aug_label, mask_param, mask_type)

def time_shifting(wav_file, output_dir, aug_label, shift_max=0.5, seed=133,
                 sample_rate=16000):
    """
    Apply time shifting augmentation to an audio file.
    
    This function is maintained for backward compatibility.
    It's recommended to use the SpecAugment class for new code.
    
    Args:
        wav_file (str): Path to the input audio file
        output_dir (str): Directory where the augmented audio will be saved
        aug_label (str): Label to prefix the output filename
        shift_max (float, optional): Maximum shift as a fraction of total length. Defaults to 0.5.
        seed (int, optional): Random seed for reproducibility. Defaults to 133.
        sample_rate (int, optional): Sample rate of the audio. Defaults to 16000.
    
    Returns:
        str: Path to the saved augmented audio file
    """
    augmenter = SpecAugment(sample_rate=sample_rate, seed=seed)
    return augmenter.time_shifting(wav_file, output_dir, aug_label, shift_max)