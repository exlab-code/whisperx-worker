"""
Clean noScribe-based RunPod worker implementation.
Architecture: Diarization → Transcription → Alignment
"""
import gc
import os
import time
import torch
from pathlib import Path
from typing import Optional

import runpod
from faster_whisper import WhisperModel, VadOptions
from pyannote.audio import Pipeline
from cog import BasePredictor, Input, Path as CogPath
from pydantic import BaseModel

# Dynamic device detection (RunPod pattern)
DEVICE = "cuda" if runpod.util.is_cuda_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"


class Output(BaseModel):
    segments: list
    language: Optional[str] = None


class Predictor(BasePredictor):
    def __init__(self):
        self.faster_whisper_model = None
        self.diarization_pipeline = None
        
    def setup(self):
        """Initialize models once during container startup"""
        print(f"🖥️  Device: {DEVICE}, Compute type: {COMPUTE_TYPE}")
        
        # Load faster-whisper model (like noScribe)
        print("Loading faster-whisper model...")
        start_time = time.time()
        self.faster_whisper_model = WhisperModel(
            "large-v3",  # Use HuggingFace model ID
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            local_files_only=False
        )
        elapsed = (time.time() - start_time) * 1000
        print(f"✅ faster-whisper model loaded: {elapsed:.2f}ms")
        
        # Load diarization pipeline (like noScribe)
        print("Loading pyannote.audio diarization pipeline...")
        start_time = time.time()
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1"
            # Note: HuggingFace token will be required via environment variable
        )
        
        if DEVICE == "cuda":
            self.diarization_pipeline = self.diarization_pipeline.to(torch.device(DEVICE))
            
        elapsed = (time.time() - start_time) * 1000
        print(f"✅ Diarization pipeline loaded: {elapsed:.2f}ms")
        
        if DEVICE == "cuda":
            print(f"📊 GPU Memory after setup: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")

    def predict(
        self,
        audio_file: CogPath = Input(description="Audio file to process"),
        language: Optional[str] = Input(description="Language code (e.g., 'en', 'de'). Auto-detect if None", default=None),
        num_speakers: Optional[int] = Input(description="Number of speakers (None for auto-detection)", default=None),
        enable_diarization: bool = Input(description="Enable speaker diarization", default=True),
        speech_pad_ms: int = Input(description="Speech padding in milliseconds", default=400),
    ) -> Output:
        """Process audio following noScribe's exact architecture"""
        
        if DEVICE == "cuda":
            print(f"📊 GPU Memory before processing: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        try:
            # ==========================================
            # PHASE 1: DIARIZATION FIRST (like noScribe line 1171)
            # ==========================================
            diarization_segments = []
            
            if enable_diarization:
                print("🔄 Phase 1: Running diarization...")
                start_time = time.time()
                
                # Run diarization on full audio file (like noScribe diarize.py)
                diarization_params = {}
                if num_speakers is not None:
                    diarization_params["num_speakers"] = num_speakers
                    
                diarization = self.diarization_pipeline(str(audio_file), **diarization_params)
                
                # Convert pyannote Annotation to list format
                for segment, _, speaker in diarization.itertracks(yield_label=True):
                    diarization_segments.append({
                        "start": segment.start,
                        "end": segment.end, 
                        "speaker": speaker
                    })
                
                elapsed = (time.time() - start_time) * 1000
                print(f"✅ Diarization complete: {len(diarization_segments)} segments, {elapsed:.2f}ms")
                
                # Explicit memory cleanup (like noScribe line 1452)
                del diarization
                gc.collect()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            
            # ==========================================  
            # PHASE 2: TRANSCRIPTION SECOND (like noScribe line 1425)
            # ==========================================
            print("🔄 Phase 2: Running transcription...")
            start_time = time.time()
            
            # Language detection if needed
            detected_language = language
            if language is None:
                print("🔍 Auto-detecting language...")
                language_info = self.faster_whisper_model.detect_language(str(audio_file))
                detected_language = language_info[0]
                print(f"✅ Detected language: {detected_language} (confidence: {language_info[1]:.3f})")
            
            # Create VAD options with padding (like noScribe line 1430)
            vad_parameters = VadOptions(speech_pad_ms=speech_pad_ms)
            print(f"🎯 Using VAD padding: {speech_pad_ms}ms")
            
            # Transcribe with VAD filtering (like noScribe lines 1455-1467)
            segments, info = self.faster_whisper_model.transcribe(
                str(audio_file),
                language=detected_language,
                beam_size=5,  # Match noScribe
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=vad_parameters
            )
            
            # Convert to list format
            transcription_segments = []
            for segment in segments:
                transcription_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": [
                        {"start": word.start, "end": word.end, "word": word.word}
                        for word in segment.words
                    ] if segment.words else []
                })
            
            elapsed = (time.time() - start_time) * 1000
            print(f"✅ Transcription complete: {len(transcription_segments)} segments, {elapsed:.2f}ms")
            
            # ==========================================
            # PHASE 3: ALIGNMENT (like noScribe lines 1533-1581)
            # ==========================================
            final_segments = transcription_segments
            
            if enable_diarization and diarization_segments:
                print("🔄 Phase 3: Aligning speakers to transcription...")
                start_time = time.time()
                
                # Align transcription segments with speaker segments
                final_segments = self._align_speakers_to_segments(
                    transcription_segments, 
                    diarization_segments
                )
                
                elapsed = (time.time() - start_time) * 1000
                print(f"✅ Speaker alignment complete: {elapsed:.2f}ms")
            
            # Final memory cleanup
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                print(f"📊 GPU Memory after processing: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            
            return Output(
                segments=final_segments,
                language=detected_language
            )
            
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            # Cleanup on error
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            raise
    
    def _align_speakers_to_segments(self, transcription_segments, diarization_segments):
        """Align transcribed segments with speaker segments using overlap detection"""
        aligned_segments = []
        
        for trans_seg in transcription_segments:
            trans_start = trans_seg["start"]
            trans_end = trans_seg["end"]
            
            # Find best matching speaker segment (like noScribe find_speaker function)
            best_speaker = "SPEAKER_00"  # Default
            best_overlap = 0
            
            for diar_seg in diarization_segments:
                # Calculate overlap percentage
                overlap_start = max(trans_start, diar_seg["start"])
                overlap_end = min(trans_end, diar_seg["end"])
                
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    trans_duration = trans_end - trans_start
                    overlap_percentage = overlap_duration / trans_duration if trans_duration > 0 else 0
                    
                    # Use speaker with highest overlap (noScribe uses 80% threshold)
                    if overlap_percentage > best_overlap and overlap_percentage >= 0.5:
                        best_overlap = overlap_percentage
                        best_speaker = diar_seg["speaker"]
            
            # Add speaker to segment
            aligned_segment = trans_seg.copy()
            aligned_segment["speaker"] = best_speaker
            aligned_segments.append(aligned_segment)
        
        return aligned_segments