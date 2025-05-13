import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt


def visualize_spectrogram(
    audio_path,
    ax=None,
    title=None,
    sample_rate=None,
    n_fft=2048,
    n_mels=128,
    hop_length=512,
    win_length=2048,
    f_min=0,
    f_max=None,
    cmap='viridis',
    vmin=None,
    vmax=None,
    figsize=(10, 4),
    colorbar=True,
    show_axis=True,
    show=True,
    return_spectrogram=False,
    db_scale=True,
    top_db=80
):
    """
    Visualize the spectrogram of an audio file.
    
    This function loads an audio file and generates a spectrogram visualization.
    It can be used both standalone and as part of a larger figure with subplots.
    
    Args:
        audio_path (str): Path to the audio file.
        ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on. If None, creates a new figure.
        title (str, optional): Title for the plot. If None, uses the filename.
        sample_rate (int, optional): Sample rate to resample audio to. If None, uses the file's sample rate.
        n_fft (int, optional): Size of FFT. Defaults to 2048.
        n_mels (int, optional): Number of mel filterbanks. Defaults to 128.
        hop_length (int, optional): Length of hop between STFT windows. Defaults to 512.
        win_length (int, optional): Window size for STFT. Defaults to 2048.
        f_min (float, optional): Minimum frequency. Defaults to 0.
        f_max (float, optional): Maximum frequency. If None, uses half the sample rate.
        cmap (str or matplotlib.colors.Colormap, optional): Colormap for the spectrogram. Defaults to 'viridis'.
        vmin (float, optional): Minimum value for colormap normalization. If None, auto-scaled.
        vmax (float, optional): Maximum value for colormap normalization. If None, auto-scaled.
        figsize (tuple, optional): Figure size (width, height) in inches. Defaults to (10, 4).
        colorbar (bool, optional): Whether to display a colorbar. Defaults to True.
        show_axis (bool, optional): Whether to show axis labels and ticks. Defaults to True.
        show (bool, optional): Whether to display the plot. Defaults to True.
        return_spectrogram (bool, optional): Whether to return the computed spectrogram. Defaults to False.
        db_scale (bool, optional): Whether to convert to decibel scale. Defaults to True.
        top_db (float, optional): Top dB to use for amplitude to dB conversion. Defaults to 80.
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects.
        If return_spectrogram is True, returns (fig, ax, spectrogram).
    """
    # Create figure if no axis is provided
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        standalone = False
    
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if needed
    if sample_rate is not None and sr != sample_rate:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        resampler = torchaudio.transforms.Resample(sr, sample_rate).to(device)
        waveform = resampler(waveform.to(device)).cpu()
        sr = sample_rate
    
    # Ensure mono if needed (take first channel)
    if waveform.shape[0] > 1:
        waveform = waveform[0].unsqueeze(0)
    
    # Set f_max if not provided
    if f_max is None:
        f_max = sr // 2
    
    # Generate mel spectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length,
        f_min=f_min,
        f_max=f_max,
    )(waveform)
    
    # Convert to decibels scale if requested
    if db_scale:
        mel_spectrogram = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(mel_spectrogram)
    
    # Convert to numpy for plotting
    spectrogram_np = mel_spectrogram.squeeze().numpy()
    
    # Plot spectrogram
    img = ax.imshow(
        spectrogram_np, 
        origin='lower',
        aspect='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[0, waveform.shape[1]/sr, f_min, f_max]
    )
    
    # Add colorbar if requested
    if colorbar:
        if standalone:
            fig.colorbar(img, ax=ax, format='%+2.0f dB' if db_scale else '%+2.0f')
        else:
            plt.gcf().colorbar(img, ax=ax, format='%+2.0f dB' if db_scale else '%+2.0f')
    
    # Set title if provided, otherwise use filename
    if title is None:
        title = os.path.basename(audio_path)
    ax.set_title(title)
    
    # Set axis labels and ticks
    if show_axis:
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Show plot if standalone
    if show and standalone:
        plt.tight_layout()
        plt.show()
    
    # Return results
    if return_spectrogram:
        return fig, ax, spectrogram_np
    else:
        return fig, ax


def compare_spectrograms(
    original_path, 
    augmented_path, 
    titles=None, 
    figsize=(15, 5),
    **kwargs
):
    """
    Compare the spectrograms of original and augmented audio files side by side.
    
    Args:
        original_path (str): Path to the original audio file.
        augmented_path (str): Path to the augmented audio file.
        titles (list, optional): List of titles for the plots. If None, uses filenames.
        figsize (tuple, optional): Figure size (width, height) in inches. Defaults to (15, 5).
        **kwargs: Additional arguments to pass to visualize_spectrogram.
    
    Returns:
        tuple: (fig, axes) matplotlib figure and axes objects.
    """
    # Create subplot figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Set default titles if not provided
    if titles is None:
        titles = [os.path.basename(original_path), os.path.basename(augmented_path)]
    
    # Plot original spectrogram
    visualize_spectrogram(
        original_path, 
        ax=axes[0], 
        title=titles[0], 
        show=False, 
        **kwargs
    )
    
    # Plot augmented spectrogram
    visualize_spectrogram(
        augmented_path, 
        ax=axes[1], 
        title=titles[1], 
        show=False, 
        **kwargs
    )
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    return fig, axes


def compare_multiple_augmentations(
    original_path, 
    augmented_paths, 
    titles=None, 
    ncols=2,
    figsize=None,
    **kwargs
):
    """
    Compare the spectrograms of original and multiple augmented audio files.
    
    Args:
        original_path (str): Path to the original audio file.
        augmented_paths (list): List of paths to the augmented audio files.
        titles (list, optional): List of titles for the plots. If None, uses filenames.
        ncols (int, optional): Number of columns in the grid. Defaults to 2.
        figsize (tuple, optional): Figure size (width, height) in inches. If None, auto-calculated.
        **kwargs: Additional arguments to pass to visualize_spectrogram.
    
    Returns:
        tuple: (fig, axes) matplotlib figure and axes objects.
    """
    # Calculate number of rows/columns
    total_plots = len(augmented_paths) + 1  # +1 for the original
    nrows = (total_plots + ncols - 1) // ncols  # Ceiling division
    
    # Set default figure size if not provided
    if figsize is None:
        figsize = (5 * ncols, 3 * nrows)
    
    # Create subplot figure
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Convert axes to flattened array for easier indexing
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()
    
    # Set default titles if not provided
    if titles is None:
        titles = [os.path.basename(original_path)] + [os.path.basename(path) for path in augmented_paths]
    
    # Plot original spectrogram
    visualize_spectrogram(
        original_path, 
        ax=axes_flat[0], 
        title=titles[0], 
        show=False, 
        **kwargs
    )
    
    # Plot augmented spectrograms
    for i, path in enumerate(augmented_paths):
        visualize_spectrogram(
            path, 
            ax=axes_flat[i+1], 
            title=titles[i+1], 
            show=False, 
            **kwargs
        )
    
    # Hide empty subplots if any
    for i in range(total_plots, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    return fig, axes
