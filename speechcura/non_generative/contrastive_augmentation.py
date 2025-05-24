import random
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from itertools import product
from pydub import AudioSegment
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class ContrastiveAugmentation:
    """
    Generates augmented samples by masking words in audio and transcription.
    Preserves the original sample rate of each audio file.
    """

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initializes the ContrastiveAugmentation class.
        
        Args:
            max_workers (Optional[int]): Maximum number of worker threads for parallel processing.
                                       Defaults to min(32, (os.cpu_count() or 1) + 4).
        """
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)

    def generate_negative_samples(
        self,
        audio_path: str,
        transcription_timestamps: List[Dict[str, Any]], # {'text': str, 'start': float (seconds), 'end': float (seconds)}
        n_aug: int,
        n_neg: int,
        p: float, # Percentage of words to mask, e.g., 0.1 for 10%
        output_dir: str,
    ) -> Dict[str, str]:
        """
        Generates augmented audio samples and their masked transcriptions.

        For each of the (n_aug * n_neg) samples:
        1. Randomly selects 'p' percentage of words from the transcription.
        2. Replaces selected words in the transcription with '[MASK]'.
        3. Silences the corresponding segments in the audio.
        4. Saves the modified audio to 'output_dir' with original sample rate.

        Args:
            audio_path (str): Path to the original audio file.
            transcription_timestamps (List[Dict[str, Any]]): A list of dictionaries,
                where each dictionary represents a word and has 'text' (str),
                'start' (float, in seconds), and 'end' (float, in seconds) keys.
            n_aug (int): Number of augmentation rounds.
            n_neg (int): Number of negative samples to generate in each round.
            p (float): The percentage of words to mask (value between 0.0 and 1.0).
            output_dir (str): Directory where augmented audio files will be saved.
                               Filenames will be like 'originalStem_augX_negY.ext'.

        Returns:
            Dict[str, str]: A dictionary where keys are the generated filenames
                            (relative to output_dir, e.g., 'originalStem_aug0_neg0.wav')
                            and values are their corresponding masked transcription strings.
                            If audio export fails, the value will be an error message.
        
        Raises:
            ValueError: If 'p' is not between 0.0 and 1.0.
            FileNotFoundError: If the audio_path does not exist.
            RuntimeError: If there's an error loading or processing the audio file.
        """
        if not (0.0 <= p <= 1.0):
            logger.error(f"Percentage 'p' must be between 0.0 and 1.0, got {p}")
            raise ValueError("Percentage 'p' must be between 0.0 and 1.0.")
        if n_aug <= 0 or n_neg <= 0:
            logger.info("n_aug or n_neg is zero or negative, no augmentations will be generated.")
            return {}

        try:
            original_audio = AudioSegment.from_file(audio_path)
        except FileNotFoundError:
            logger.error(f"Audio file not found: {audio_path}")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        except Exception as e: 
            logger.error(f"Error loading audio file {audio_path}. Original error: {e}", exc_info=True)
            raise RuntimeError(
                f"Error loading audio file {audio_path}. "
                f"Ensure FFmpeg/Libav is installed and accessible by pydub. Original error: {e}"
            )

        logger.info(f"Processing {audio_path} with original sample rate: {original_audio.frame_rate}Hz")

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        all_masked_samples: Dict[str, str] = {}
        
        original_audio_path_obj = Path(audio_path)
        original_audio_stem = original_audio_path_obj.stem
        original_audio_suffix = original_audio_path_obj.suffix 
        audio_export_format = original_audio_suffix[1:] if original_audio_suffix and original_audio_suffix.startswith('.') else "wav"

        num_total_words = len(transcription_timestamps)
        num_words_to_mask_target = int(round(num_total_words * p))
        
        # Convert audio to numpy array for faster processing
        audio_array = np.array(original_audio.get_array_of_samples())
        if original_audio.channels == 2:
            audio_array = audio_array.reshape((-1, 2))
        
        # Pre-calculate all mask combinations for batch processing
        all_mask_combinations = []
        for aug_idx in range(n_aug):
            for neg_idx in range(n_neg):
                if num_total_words > 0 and num_words_to_mask_target > 0:
                    actual_num_to_mask = min(num_words_to_mask_target, num_total_words)
                    word_indices = list(range(num_total_words))
                    indices_to_mask = random.sample(word_indices, actual_num_to_mask)
                else:
                    indices_to_mask = []
                all_mask_combinations.append((aug_idx, neg_idx, indices_to_mask))

        # Process all combinations in parallel
        total_iterations = len(all_mask_combinations)
        if total_iterations == 0:
            return {}
        
        def process_single_sample(mask_data):
            aug_idx, neg_idx, indices_to_mask = mask_data
            current_transcription_texts = [item['text'] for item in transcription_timestamps]
            
            # Create masked audio using numpy operations (much faster)
            masked_audio_array = audio_array.copy()
            
            if indices_to_mask:
                # Mark transcription
                for idx_to_mask in indices_to_mask:
                    current_transcription_texts[idx_to_mask] = "[MASK]"
                
                # Silence audio segments using vectorized operations
                for idx_to_mask in indices_to_mask:
                    word_info = transcription_timestamps[idx_to_mask]
                    start_s = float(word_info['start'])
                    end_s = float(word_info['end'])
                    
                    # Convert to sample indices
                    start_sample = int(start_s * original_audio.frame_rate)
                    end_sample = int(end_s * original_audio.frame_rate)
                    
                    # Ensure bounds
                    start_sample = max(0, start_sample)
                    end_sample = min(len(masked_audio_array), end_sample)
                    
                    if start_sample < end_sample:
                        # Zero out the audio segment (silence)
                        if original_audio.channels == 2:
                            masked_audio_array[start_sample:end_sample, :] = 0
                        else:
                            masked_audio_array[start_sample:end_sample] = 0
            
            # Convert back to AudioSegment
            if original_audio.channels == 2:
                final_audio_array = masked_audio_array.flatten()
            else:
                final_audio_array = masked_audio_array
            
            final_masked_audio = AudioSegment(
                final_audio_array.tobytes(),
                frame_rate=original_audio.frame_rate,
                sample_width=original_audio.sample_width,
                channels=original_audio.channels
            )
            
            masked_transcription_string = " ".join(current_transcription_texts)
            output_filename = f"{original_audio_stem}_aug{aug_idx}_neg{neg_idx}{original_audio_suffix or '.' + audio_export_format}"
            output_filepath = output_dir_path / output_filename
            
            try:
                # Use faster export parameters
                final_masked_audio.export(
                    str(output_filepath), 
                    format=audio_export_format,
                    parameters=["-q:a", "0"]  # Fastest encoding
                )
                return output_filename, masked_transcription_string
            except Exception as e:
                error_msg = (f"[ERROR: Audio export failed for sample {output_filename}. "
                             f"Transcription was: '{masked_transcription_string}'. Error: {e}]")
                logger.warning(f"Error exporting augmented audio file {output_filepath}: {e}")
                return output_filename, error_msg
        
        # Process samples in parallel with progress bar
        progress_bar_desc = f"Generating samples for {original_audio_stem}"
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_data = {executor.submit(process_single_sample, mask_data): mask_data 
                             for mask_data in all_mask_combinations}
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_data), 
                             total=total_iterations, 
                             desc=progress_bar_desc, 
                             unit="sample", 
                             leave=False):
                try:
                    filename, transcription = future.result()
                    all_masked_samples[filename] = transcription
                except Exception as e:
                    mask_data = future_to_data[future]
                    aug_idx, neg_idx, _ = mask_data
                    error_filename = f"{original_audio_stem}_aug{aug_idx}_neg{neg_idx}_ERROR.txt"
                    all_masked_samples[error_filename] = f"Processing error: {e}"
                    logger.error(f"Error in parallel processing: {e}")

        return all_masked_samples

    def augment_directory(
        self,
        input_audio_dir: str,
        input_transcriptions_dir: str,
        n_aug: int,
        n_neg: int,
        p: float,
        main_output_dir: str,
        audio_extensions: Optional[List[str]] = None,
        transcription_extension: str = ".json"
    ) -> Dict[str, str]:
        """
        Performs generate_negative_samples on all audio files in a directory.

        Args:
            input_audio_dir (str): Directory containing original audio files.
            input_transcriptions_dir (str): Directory containing corresponding
                                           transcription files. Assumes transcription
                                           files have the same stem as audio files but
                                           with 'transcription_extension'.
            n_aug (int): Number of augmentation rounds for each file.
            n_neg (int): Number of negative samples per round for each file.
            p (float): Percentage of words to mask for each file.
            main_output_dir (str): Main directory where augmented audio files will be saved.
                                   Augmented files are saved directly in this directory.
            audio_extensions (Optional[List[str]]): List of audio file extensions to process
                                                   (e.g., ['.wav', '.mp3']). Defaults to
                                                   ['.wav', '.mp3', '.flac'].
            transcription_extension (str): Extension for transcription files (e.g., '.json').
                                          Defaults to '.json'.

        Returns:
            Dict[str, str]: An aggregated dictionary where keys are the
                            generated filenames (e.g., 'originalAudioStem_aug0_neg0.wav')
                            directly within 'main_output_dir', and values are their
                            corresponding masked transcriptions or error messages.
        """
        if audio_extensions is None:
            audio_extensions = ['.wav', '.mp3', '.flac']
        
        input_audio_path = Path(input_audio_dir)
        input_transcriptions_path = Path(input_transcriptions_dir)
        main_output_path = Path(main_output_dir)
        main_output_path.mkdir(parents=True, exist_ok=True)

        aggregated_results: Dict[str, str] = {}

        if not input_audio_path.is_dir():
            logger.error(f"Input audio directory not found: {input_audio_dir}")
            return {}
        if not input_transcriptions_path.is_dir():
            logger.error(f"Input transcriptions directory not found: {input_transcriptions_dir}")
            return {}
        
        # Convert iterator to list for tqdm to get total count and display progress bar
        audio_files_to_process = [f for f in input_audio_path.iterdir() if f.is_file() and f.suffix.lower() in audio_extensions]
        if not audio_files_to_process:
            logger.info(f"No audio files found with extensions {audio_extensions} in {input_audio_dir}.")
            return {}

        def process_single_file(audio_file_path):
            """Process a single audio file and return results."""
            audio_stem = audio_file_path.stem
            transcription_file_name = audio_stem + transcription_extension
            transcription_file_path = input_transcriptions_path / transcription_file_name

            if not transcription_file_path.exists():
                logger.warning(f"Transcription file not found for {audio_file_path.name} at {transcription_file_path}, skipping.")
                return {}
            
            try:
                with open(transcription_file_path, 'r', encoding='utf-8') as f:
                    transcription_timestamps = json.load(f)
                if not isinstance(transcription_timestamps, list) or \
                   not all(isinstance(item, dict) and 'text' in item and 'start' in item and 'end' in item 
                           for item in transcription_timestamps):
                    logger.warning(f"Invalid transcription format in {transcription_file_path} for {audio_file_path.name}. Skipping.")
                    return {}
            except json.JSONDecodeError:
                logger.warning(f"Error decoding JSON from {transcription_file_path} for {audio_file_path.name}. Skipping.")
                return {}
            except Exception as e:
                logger.warning(f"Error reading transcription file {transcription_file_path} for {audio_file_path.name}: {e}. Skipping.")
                return {}

            logger.info(f"Processing {audio_file_path.name}...")
            try:
                current_file_results = self.generate_negative_samples(
                    audio_path=str(audio_file_path),
                    transcription_timestamps=transcription_timestamps,
                    n_aug=n_aug,
                    n_neg=n_neg,
                    p=p,
                    output_dir=str(main_output_path)
                )
                return current_file_results
            except Exception as e:
                logger.error(f"Error processing {audio_file_path.name}: {e}")
                error_key = f"PROC_ERROR_{audio_file_path.name}.txt"
                return {error_key: f"Error processing {audio_file_path.name}: {e}"}

        # Process files in parallel
        with ThreadPoolExecutor(max_workers=min(4, self.max_workers)) as executor:  # Limit file-level parallelism
            # Submit all file processing tasks
            future_to_file = {executor.submit(process_single_file, audio_file): audio_file 
                             for audio_file in audio_files_to_process}
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_file), 
                             total=len(audio_files_to_process),
                             desc=f"Augmenting directory {input_audio_dir}", 
                             unit="file"):
                try:
                    file_results = future.result()
                    aggregated_results.update(file_results)
                except Exception as e:
                    audio_file = future_to_file[future]
                    error_key = f"PROC_ERROR_{audio_file.name}.txt"
                    aggregated_results[error_key] = f"Error processing {audio_file.name}: {e}"
                    logger.error(f"Error in file processing: {e}")

        return aggregated_results
