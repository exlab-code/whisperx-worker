from cog import BasePredictor, Input, Path, BaseModel
from pydub import AudioSegment
from typing import Any
from whisperx.audio import N_SAMPLES, log_mel_spectrogram
from pyannote_diarizer import run_diarization

import gc
import math
import os
import shutil
import whisperx
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions
import tempfile
import time
import torch
import runpod

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# Dynamic compute type based on device (matches official RunPod worker pattern)
# Will be set in setup() based on actual device availability
whisper_arch = "large-v3"


class Output(BaseModel):
    segments: Any
    detected_language: str


class Predictor(BasePredictor):
    def __init__(self):
        super().__init__()
        self.whisperx_model = None  # Initialize as None for lazy loading
        self.faster_whisper_model = None  # Initialize faster-whisper model
        self.device = None  # Add device attribute to store device string
    
    def setup(self):
        # Copy VAD model files
        source_folder = './models/vad'
        destination_folder = '../root/.cache/torch'
        file_name = 'whisperx-vad-segmentation.bin'

        os.makedirs(destination_folder, exist_ok=True)

        source_file_path = os.path.join(source_folder, file_name)
        if os.path.exists(source_file_path):
            destination_file_path = os.path.join(destination_folder, file_name)

            if not os.path.exists(destination_file_path):
                shutil.copy(source_file_path, destination_folder)
        
        # Determine the correct device
        # Use RunPod's device detection pattern (matches official worker)
        self.device = "cuda" if runpod.util.is_cuda_available() else "cpu"
        
        # Dynamic compute type based on device (matches official RunPod worker)
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        print(f"🖥️  Device: {self.device}, Compute type: {self.compute_type}")
        
        print("Loading WhisperX model during setup (one-time load for massive speedup)...")
        
        # GPU STATUS CHECK
        if self.device == "cuda":
            print(f"✅ CUDA available: {torch.cuda.device_count()} GPU(s)")
            print(f"✅ Current GPU: {torch.cuda.get_device_name()}")
            print(f"✅ GPU Memory before model loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")
        else:
            print("❌ CUDA NOT AVAILABLE - Models will run on CPU (VERY SLOW!)")
            
        setup_start_time = time.time_ns() / 1e6
        
        asr_options = {
            "temperatures": [0],
            "beam_size": 5,
        }
        
        # Configure VAD options - remove speech_pad as it's not valid for load_model
        vad_options = {
            "vad_onset": 0.15,    # Default sensitive values
            "vad_offset": 0.25    # Will be overridden by payload
        }
        
        # Use faster-whisper directly like noScribe for proven VAD support
        self.faster_whisper_model = WhisperModel(
            whisper_arch,
            device=self.device,
            compute_type=self.compute_type,
            local_files_only=False  # Download model from HuggingFace
        )
        
        # Note: WhisperX model will be loaded lazily only if alignment is needed
        # This saves memory when only doing transcription + diarization
        
        setup_elapsed = time.time_ns() / 1e6 - setup_start_time
        print(f"faster-whisper model loaded in setup: {setup_elapsed:.2f} ms")
        
        # VERIFY GPU USAGE AFTER MODEL LOADING
        if self.device == "cuda":
            print(f"✅ GPU Memory after faster-whisper loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")
            print(f"✅ faster-whisper model loaded on device: {self.device}")
        
        print("🚀 Subsequent requests will use cached model - expect 5-10x speedup!")

    def predict(
            self,
            audio_file: Path = Input(description="Audio file"),
            language: str = Input(
                description="ISO code of the language spoken in the audio, specify None to perform language detection",
                default=None),
            language_detection_min_prob: float = Input(
                description="If language is not specified, then the language will be detected recursively on different "
                            "parts of the file until it reaches the given probability",
                default=0
            ),
            language_detection_max_tries: int = Input(
                description="If language is not specified, then the language will be detected following the logic of "
                            "language_detection_min_prob parameter, but will stop after the given max retries. If max "
                            "retries is reached, the most probable language is kept.",
                default=5
            ),
            initial_prompt: str = Input(
                description="Optional text to provide as a prompt for the first window",
                default=None),
            batch_size: int = Input(
                description="Parallelization of input audio transcription",
                default=64),
            temperature: float = Input(
                description="Temperature to use for sampling",
                default=0),
            vad_onset: float = Input(
                description="VAD onset",
                default=0.500),
            vad_offset: float = Input(
                description="VAD offset", 
                default=0.363),
            speech_pad_seconds: float = Input(
                description="Seconds of padding around each speech segment to prevent content cutoff",
                default=0.4),
            align_output: bool = Input(
                description="Aligns whisper output to get accurate word-level timestamps",
                default=False),
            diarization: bool = Input(
                description="Assign speaker ID labels",
                default=False),
            huggingface_access_token: str = Input(
                description="To enable diarization, please enter your HuggingFace token (read). You need to accept "
                            "the user agreement for the models specified in the README.",
                default=None),
            min_speakers: int = Input(
                description="Minimum number of speakers if diarization is activated (leave blank if unknown)",
                default=None),
            max_speakers: int = Input(
                description="Maximum number of speakers if diarization is activated (leave blank if unknown)",
                default=None),
            debug: bool = Input(
                description="Print out compute/inference times and memory usage information",
                default=False)
    ) -> Output:
        with torch.inference_mode():
            # ... (The first part of the predict method is unchanged) ...
            asr_options = {
                "temperatures": [temperature],
                "initial_prompt": initial_prompt
            }

            vad_options = {
                "vad_onset": vad_onset,
                "vad_offset": vad_offset
            }

            audio_duration = get_audio_duration(audio_file)

            # Language detection if needed
            if language is None:
                print("🔍 Auto-detecting language...")
                try:
                    # Use faster-whisper for language detection like noScribe
                    faster_model = self.faster_whisper_model
                    language_info = faster_model.detect_language(audio_file)
                    language = language_info[0]  # Get detected language code
                    language_probability = language_info[1]  # Get confidence
                    print(f"✅ Detected language: {language} (confidence: {language_probability:.3f})")
                    
                    # Check if confidence meets minimum threshold
                    if language_detection_min_prob > 0 and language_probability < language_detection_min_prob:
                        print(f"⚠️ Language detection confidence {language_probability:.3f} below threshold {language_detection_min_prob}")
                        # Continue anyway but warn user
                        
                except Exception as e:
                    print(f"❌ Language detection failed: {e}")
                    language = "en"  # Fallback to English
                    print(f"🔄 Falling back to English")

            start_time = time.time_ns() / 1e6

            # VERIFY GPU USAGE DURING PREDICTION  
            if self.device == "cuda":
                print(f"📊 GPU Memory before transcription: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
                print(f"📊 Model device check: {self.device}")

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to load model: {elapsed_time:.2f} ms")

            start_time = time.time_ns() / 1e6

            # ========== EXACT NOSCRIBE APPROACH: faster-whisper with VAD ==========
            
            # Fix FieldInfo parameter access issue
            pad_seconds = float(speech_pad_seconds)
            speech_pad_ms = int(pad_seconds * 1000)
            
            # Create VadOptions exactly like noScribe
            vad_parameters = VadOptions(
                speech_pad_ms=speech_pad_ms,  # 400ms padding like noScribe
            )
            
            print(f"🔄 Using noScribe approach: faster-whisper with VadOptions(speech_pad_ms={speech_pad_ms})")
            
            # Use faster-whisper model directly like noScribe
            faster_model = self.faster_whisper_model
            
            # Transcribe with exact noScribe parameters  
            segments, info = faster_model.transcribe(
                audio_file,  # Use audio file path like noScribe
                language=language,
                beam_size=5,  # Match noScribe's beam_size
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=vad_parameters
            )
            
            # Convert faster-whisper segments to WhisperX-compatible format
            segments_list = []
            for segment in segments:
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                }
                # Add word-level timestamps if available
                if hasattr(segment, 'words') and segment.words:
                    segment_dict["words"] = [
                        {
                            "start": word.start,
                            "end": word.end,
                            "word": word.word
                        }
                        for word in segment.words
                    ]
                segments_list.append(segment_dict)
            
            result = {
                "segments": segments_list,
                "language": info.language
            }
            detected_language = info.language
            print(f"✅ Transcribed with faster-whisper VAD: {len(result['segments'])} segments detected")
            
            # Apply speaker diarization if requested
            if diarization:
                print("🔄 Applying speaker diarization to transcribed segments")
                
                # Load audio for diarization (needed by pyannote)
                audio = whisperx.load_audio(audio_file)
                speaker_segments = get_diarization_segments(audio, debug, huggingface_access_token, min_speakers, max_speakers, self.device)
                print(f"✅ Diarization complete: {len(speaker_segments)} speaker segments")
                
                # Assign speakers to transcribed segments using whisperx.assign_word_speakers
                try:
                    result = whisperx.assign_word_speakers(speaker_segments, result)
                    print("✅ Speaker assignment complete")
                except Exception as e:
                    print(f"❌ Error during speaker assignment: {e}")
                    # Continue without speaker assignment

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to transcribe: {elapsed_time:.2f} ms")

            gc.collect()
            torch.cuda.empty_cache()

            if align_output:
                if detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_TORCH or detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_HF:
                    # Load WhisperX model lazily for alignment (memory optimization)
                    if self.whisperx_model is None:
                        print("🔄 Loading WhisperX model for alignment...")
                        asr_options = {"temperatures": [0], "beam_size": 5}
                        vad_options = {"vad_onset": 0.15, "vad_offset": 0.25}
                        self.whisperx_model = whisperx.load_model(
                            whisper_arch, 
                            self.device,
                            compute_type=self.compute_type,
                            language=None,
                            asr_options=asr_options,
                            vad_options=vad_options
                        )
                        print("✅ WhisperX model loaded for alignment")
                    
                    # Load audio for alignment if not already loaded for diarization
                    if 'audio' not in locals():
                        audio = whisperx.load_audio(audio_file)
                    result = align(audio, result, debug)
                else:
                    print(f"Cannot align output as language {detected_language} is not supported for alignment")

            if debug:
                print(f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")

        return Output(
            segments=result["segments"],
            detected_language=detected_language
        )


def get_audio_duration(file_path):
    return len(AudioSegment.from_file(file_path))


def detect_language(full_audio_file_path, segments_starts, language_detection_min_prob,
                    language_detection_max_tries, asr_options, vad_options, iteration=1):
    model = whisperx.load_model(whisper_arch, device, compute_type=compute_type, asr_options=asr_options,
                                vad_options=vad_options)

    start_ms = segments_starts[iteration - 1]

    audio_segment_file_path = extract_audio_segment(full_audio_file_path, start_ms, 30000)

    audio = whisperx.load_audio(audio_segment_file_path)

    model_n_mels = model.model.feat_kwargs.get("feature_size")
    segment = log_mel_spectrogram(audio[: N_SAMPLES],
                                  n_mels=model_n_mels if model_n_mels is not None else 80,
                                  padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0])
    encoder_output = model.model.encode(segment)
    results = model.model.model.detect_language(encoder_output)
    language_token, language_probability = results[0][0]
    language = language_token[2:-2]

    print(f"Iteration {iteration} - Detected language: {language} ({language_probability:.2f})")

    audio_segment_file_path.unlink()

    gc.collect()
    torch.cuda.empty_cache()
    del model

    detected_language = {
        "language": language,
        "probability": language_probability,
        "iterations": iteration
    }

    if language_probability >= language_detection_min_prob or iteration >= language_detection_max_tries:
        return detected_language

    next_iteration_detected_language = detect_language(full_audio_file_path, segments_starts,
                                                       language_detection_min_prob, language_detection_max_tries,
                                                       asr_options, vad_options, iteration + 1)

    if next_iteration_detected_language["probability"] > detected_language["probability"]:
        return next_iteration_detected_language

    return detected_language


def extract_audio_segment(input_file_path, start_time_ms, duration_ms):
    input_file_path = Path(input_file_path) if not isinstance(input_file_path, Path) else input_file_path

    audio = AudioSegment.from_file(input_file_path)

    end_time_ms = start_time_ms + duration_ms
    extracted_segment = audio[start_time_ms:end_time_ms]

    file_extension = input_file_path.suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file_path = Path(temp_file.name)
        extracted_segment.export(temp_file_path, format=file_extension.lstrip('.'))

    return temp_file_path


def distribute_segments_equally(total_duration, segments_duration, iterations):
    available_duration = total_duration - segments_duration

    if iterations > 1:
        spacing = available_duration // (iterations - 1)
    else:
        spacing = 0

    start_times = [i * spacing for i in range(iterations)]

    if iterations > 1:
        start_times[-1] = total_duration - segments_duration

    return start_times


def align(audio, result, debug):
    start_time = time.time_ns() / 1e6

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device,
                            return_char_alignments=False)

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        print(f"Duration to align output: {elapsed_time:.2f} ms")

    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    return result


def get_diarization_segments(audio, debug, huggingface_access_token, min_speakers, max_speakers, device):
    """Get raw speaker segments from diarization (noScribe approach)"""
    start_time = time.time_ns() / 1e6

    # Use pyannote.audio 3.1 directly for better performance
    diarize_segments = run_diarization(
        audio_tensor=audio,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        auth_token=huggingface_access_token,
        device=device
    )

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        print(f"Duration to get diarization segments (pyannote 3.1): {elapsed_time:.2f} ms")

    gc.collect()
    torch.cuda.empty_cache()
    # No need to delete model - it's cached in pyannote_diarizer module

    return diarize_segments  # Return raw DataFrame with speaker segments

def diarize(audio, result, debug, huggingface_access_token, min_speakers, max_speakers):
    """Legacy diarization function for non-diarization-first pipeline"""
    start_time = time.time_ns() / 1e6

    # Use pyannote.audio 3.1 directly for better performance
    diarize_segments = run_diarization(
        audio_tensor=audio,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        auth_token=huggingface_access_token,
        device=device
    )

    result = whisperx.assign_word_speakers(diarize_segments, result)

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        print(f"Duration to diarize segments (pyannote 3.1): {elapsed_time:.2f} ms")

    gc.collect()
    torch.cuda.empty_cache()
    # No need to delete model - it's cached in pyannote_diarizer module

    return result
