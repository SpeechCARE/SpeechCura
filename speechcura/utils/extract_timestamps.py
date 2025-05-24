import os
import torch
import whisperx
import json
from tqdm import tqdm

def _load_models(model_size, device):
    """
    Helper function to load Whisper model.
    
    Args:
        model_size (str): Size of the Whisper model.
        device (str): Device to load the model on.
    
    Returns:
        model: Loaded Whisper model.
    """
    print("Loading Whisper model...")
    model = whisperx.load_model(model_size, device=device)
    print(f"Model {model_size} loaded successfully")
    return model

def _process_single_file(file_path, model, device, show_progress=True):
    """
    Helper function to process a single audio file and extract word-level timestamps.
    
    Args:
        file_path (str): Path to the audio file.
        model: Loaded Whisper model.
        device (str): Device being used.
        show_progress (bool): Whether to show progress bars.
    
    Returns:
        list: List of dictionaries with word-level timestamps, or None if error.
    """
    filename = os.path.basename(file_path)
    
    try:
        if show_progress:
            # Step 1: Transcribe the audio file
            with tqdm(total=3, desc=f"Transcribing {filename[:20]}...", leave=False) as pbar:
                result = model.transcribe(file_path)
                pbar.update(1)
                
                # Step 2: Load alignment model
                pbar.set_description(f"Loading alignment model for {result['language']}")
                align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
                pbar.update(1)
                
                # Step 3: Align the transcription with word-level timestamps
                pbar.set_description(f"Aligning timestamps for {filename[:20]}...")
                aligned_result = whisperx.align(result["segments"], align_model, metadata, file_path, device)
                pbar.update(1)
        else:
            # Process without progress bars
            result = model.transcribe(file_path)
            align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            aligned_result = whisperx.align(result["segments"], align_model, metadata, file_path, device)

        # Format the transcription into the desired structure
        transcription = []
        
        if "segments" in aligned_result:
            for segment in aligned_result["segments"]:
                if "words" in segment:
                    for word in segment["words"]:
                        if "word" in word and "start" in word and "end" in word:
                            transcription.append({
                                "text": word["word"].strip(),
                                "start": int(round(word["start"] * 1000)),
                                "end": int(round(word["end"] * 1000))
                            })

        return transcription
        
    except Exception as e:
        if show_progress:
            tqdm.write(f"Error processing {filename}: {str(e)}")
        else:
            print(f"Error processing {filename}: {str(e)}")
        return None

def extract_timestamps(input_file, output_file=None, model_size="large-v3"):
    """
    Extracts word-level timestamps from a single audio file.
    
    Args:
        input_file (str): Path to the input audio file.
        output_file (str, optional): Path to save the JSON output. If None, uses input filename with .json extension.
        model_size (str): Size of the Whisper model ("tiny", "base", "small", "medium", "large-v3").
    
    Returns:
        list: List of dictionaries with word-level timestamps, or None if error.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Validate input file
    if not os.path.isfile(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return None
    
    if not input_file.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.ogg', '.wma')):
        print(f"Error: Unsupported audio format for file '{input_file}'.")
        return None
    
    # Load model
    model = _load_models(model_size, device)
    
    # Process the file
    print(f"Processing: {os.path.basename(input_file)}")
    transcription = _process_single_file(input_file, model, device, show_progress=False)
    
    if transcription is None:
        return None
    
    # Save to file if output_file is specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(transcription, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(transcription)} words with timestamps to: {output_file}")
    else:
        # Generate default output filename
        output_file = os.path.splitext(input_file)[0] + ".json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(transcription, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(transcription)} words with timestamps to: {output_file}")
    
    return transcription

def extract_timestamps_directory(input_dir, output_dir, model_size="large-v3"):
    """
    Transcribes all audio files in a directory and saves the results as JSON files.
    Uses WhisperX for high-quality word-level transcription with precise timestamps.

    Args:
        input_dir (str): Path to the directory containing audio files.
        output_dir (str): Path to the directory where JSON files will be saved.
        model_size (str): Size of the Whisper model ("tiny", "base", "small", "medium", "large-v3").

    Returns:
        None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get list of audio files first
    audio_files = []
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.ogg', '.wma')):
            audio_files.append(filename)
    
    if not audio_files:
        print("No audio files found in the input directory.")
        return
    
    print(f"Found {len(audio_files)} audio files to process")

    # Load the Whisper model
    model = _load_models(model_size, device)

    # Process files with tqdm progress bar
    for filename in tqdm(audio_files, desc="Processing audio files", unit="file"):
        file_path = os.path.join(input_dir, filename)
        
        # Update progress bar description with current file
        tqdm.write(f"Processing: {filename}")

        # Process the file
        transcription = _process_single_file(file_path, model, device, show_progress=True)
        
        if transcription is not None:
            # Save the transcription as a JSON file
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_file_path = os.path.join(output_dir, output_filename)
            
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(transcription, f, ensure_ascii=False, indent=2)

            tqdm.write(f"Saved {len(transcription)} words with timestamps to: {output_filename}")

    print(f"\nProcessing complete! Transcriptions saved to: {output_dir}")
