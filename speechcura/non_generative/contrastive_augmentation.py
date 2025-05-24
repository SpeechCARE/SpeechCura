import random
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from itertools import product
from pydub import AudioSegment
from tqdm import tqdm

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

    def __init__(self):
        """
        Initializes the ContrastiveAugmentation class.
        """
        pass

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

        # Use itertools.product for a single tqdm progress bar over nested loops
        total_iterations = n_aug * n_neg
        if total_iterations == 0:
            return {}
            
        progress_bar_desc = f"Generating samples for {original_audio_stem}"
        
        for aug_idx, neg_idx in tqdm(list(product(range(n_aug), range(n_neg))), desc=progress_bar_desc, unit="sample", leave=False):
            current_transcription_texts = [item['text'] for item in transcription_timestamps]
            
            final_masked_audio: AudioSegment
            
            if num_total_words == 0 or num_words_to_mask_target == 0:
                final_masked_audio = original_audio[:] 
            else:
                actual_num_to_mask = min(num_words_to_mask_target, num_total_words)
                word_indices = list(range(num_total_words))
                indices_to_mask = random.sample(word_indices, actual_num_to_mask)

                for idx_to_mask in indices_to_mask:
                    current_transcription_texts[idx_to_mask] = "[MASK]"
                
                segments_to_silence_ms = []
                for idx_to_mask in sorted(indices_to_mask):
                    word_info = transcription_timestamps[idx_to_mask]
                    start_s = float(word_info['start'])
                    end_s = float(word_info['end'])
                    
                    start_ms, end_ms = int(start_s * 1000), int(end_s * 1000)
                    duration_ms = end_ms - start_ms
                    if duration_ms > 0:
                        segments_to_silence_ms.append({'start': start_ms, 'end': end_ms, 'duration': duration_ms})

                audio_pieces = []
                current_pos_ms = 0
                for seg_info in segments_to_silence_ms:
                    if seg_info['start'] > current_pos_ms:
                        audio_pieces.append(original_audio[current_pos_ms:seg_info['start']])
                    
                    silence = AudioSegment.silent(duration=seg_info['duration'], frame_rate=original_audio.frame_rate)
                    if silence.channels != original_audio.channels:
                        silence = silence.set_channels(original_audio.channels)
                    audio_pieces.append(silence)
                    
                    current_pos_ms = seg_info['end']
                
                if current_pos_ms < len(original_audio):
                    audio_pieces.append(original_audio[current_pos_ms:])
                
                if not audio_pieces: 
                    final_masked_audio = original_audio[:] if len(original_audio) > 0 else AudioSegment.silent(duration=0, frame_rate=original_audio.frame_rate).set_channels(original_audio.channels)
                else:
                    empty_sound_for_sum = AudioSegment.silent(duration=0, frame_rate=original_audio.frame_rate)
                    empty_sound_for_sum = empty_sound_for_sum.set_channels(original_audio.channels)
                    
                    built_audio = empty_sound_for_sum
                    for piece in audio_pieces:
                        built_audio += piece
                    final_masked_audio = built_audio
            
            masked_transcription_string = " ".join(current_transcription_texts)
            
            output_filename_stem = original_audio_path_obj.stem
            output_filename = f"{output_filename_stem}_aug{aug_idx}_neg{neg_idx}{original_audio_suffix or '.' + audio_export_format}"
            output_filepath = output_dir_path / output_filename
            
            try:
                final_masked_audio.export(str(output_filepath), format=audio_export_format)
                all_masked_samples[output_filename] = masked_transcription_string
            except Exception as e:
                error_msg = (f"[ERROR: Audio export failed for sample {output_filename}. "
                             f"Transcription was: '{masked_transcription_string}'. Error: {e}]")
                all_masked_samples[output_filename] = error_msg
                logger.warning(f"Error exporting augmented audio file {output_filepath}: {e}", exc_info=True)

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

        for audio_file_path in tqdm(audio_files_to_process, desc=f"Augmenting directory {input_audio_dir}", unit="file"):
            audio_stem = audio_file_path.stem
            transcription_file_name = audio_stem + transcription_extension
            transcription_file_path = input_transcriptions_path / transcription_file_name

            if not transcription_file_path.exists():
                logger.warning(f"Transcription file not found for {audio_file_path.name} at {transcription_file_path}, skipping.")
                continue
            
            try:
                with open(transcription_file_path, 'r', encoding='utf-8') as f:
                    transcription_timestamps = json.load(f)
                if not isinstance(transcription_timestamps, list) or \
                   not all(isinstance(item, dict) and 'text' in item and 'start' in item and 'end' in item 
                           for item in transcription_timestamps):
                    logger.warning(f"Invalid transcription format in {transcription_file_path} for {audio_file_path.name}. Skipping.")
                    continue
            except json.JSONDecodeError:
                logger.warning(f"Error decoding JSON from {transcription_file_path} for {audio_file_path.name}. Skipping.", exc_info=True)
                continue
            except Exception as e:
                logger.warning(f"Error reading transcription file {transcription_file_path} for {audio_file_path.name}: {e}. Skipping.", exc_info=True)
                continue

            # Augmented files will be saved directly in main_output_dir
            # per_file_output_dir = main_output_path / audio_stem
            # per_file_output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing {audio_file_path.name}...")
            try:
                current_file_results = self.generate_negative_samples(
                    audio_path=str(audio_file_path),
                    transcription_timestamps=transcription_timestamps,
                    n_aug=n_aug,
                    n_neg=n_neg,
                    p=p,
                    output_dir=str(main_output_path) # Save directly to main_output_dir
                )
                
                for gen_filename, transcript in current_file_results.items():
                    # Key in aggregated_results is now just the filename, relative to main_output_dir
                    # relative_path_key = str(Path(audio_stem) / gen_filename) 
                    aggregated_results[gen_filename] = transcript
            except Exception as e:
                logger.error(f"Error processing {audio_file_path.name}: {e}", exc_info=True)
                # Log error with a unique key if processing for the file fails entirely
                error_key = f"PROC_ERROR_{audio_file_path.name}.txt"
                aggregated_results[error_key] = f"Error processing {audio_file_path.name}: {e}"

        return aggregated_results
