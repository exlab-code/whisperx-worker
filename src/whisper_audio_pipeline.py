#!/usr/bin/env python3
"""
Enhanced Audio Processing Pipeline for Whisper
Implements best practices for audio preprocessing before transcription
"""

import numpy as np
import librosa
import soundfile as sf
import torch
from pathlib import Path
import logging
from typing import Tuple, List, Optional, Dict, Any
import tempfile
import os

# VAD imports
try:
    import torch
    import torchaudio
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
    SILERO_VAD_AVAILABLE = True
except ImportError:
    SILERO_VAD_AVAILABLE = False
    print("Silero VAD not available - VAD features disabled")

# Source separation imports
try:
    from demucs import pretrained
    from demucs.apply import apply_model
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False
    print("Demucs not available - source separation disabled")

logger = logging.getLogger(__name__)


class WhisperAudioPipeline:
    """Enhanced audio processing pipeline optimized for Whisper transcription"""
    
    def __init__(self, 
                 target_sr: int = 16000,
                 enable_vad: bool = True,
                 enable_source_separation: bool = False,
                 segment_length: float = 30.0):
        """
        Initialize the audio processing pipeline
        
        Args:
            target_sr: Target sample rate (16kHz recommended for Whisper)
            enable_vad: Enable voice activity detection
            enable_source_separation: Enable source separation with Demucs
            segment_length: Target segment length in seconds
        """
        self.target_sr = target_sr
        self.enable_vad = enable_vad and SILERO_VAD_AVAILABLE
        self.enable_source_separation = enable_source_separation and DEMUCS_AVAILABLE
        self.segment_length = segment_length
        
        # Initialize models
        self.vad_model = None
        self.demucs_model = None
        
        self._init_models()
    
    def _init_models(self):
        """Initialize audio processing models"""
        # Initialize VAD model
        if self.enable_vad:
            try:
                self.vad_model = load_silero_vad()
                logger.info("Silero VAD model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load VAD model: {e}")
                self.enable_vad = False
        
        # Initialize Demucs model
        if self.enable_source_separation:
            try:
                self.demucs_model = pretrained.get_model("htdemucs")
                logger.info("Demucs model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Demucs model: {e}")
                self.enable_source_separation = False
    
    def basic_audio_cleanup(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """
        Basic audio cleanup: normalize, remove DC offset, convert format
        
        Args:
            audio: Input audio array
            sr: Sample rate
            
        Returns:
            Tuple of (cleaned_audio, sample_rate)
        """
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Normalize volume levels (prevent clipping)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95
        
        # Convert to target sample rate
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
        
        # Ensure mono
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        # Convert to 16-bit equivalent range but keep float32
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio, sr
    
    def detect_voice_activity(self, audio: np.ndarray, sr: int) -> List[Dict[str, float]]:
        """
        Detect voice activity using Silero VAD
        
        Args:
            audio: Input audio array
            sr: Sample rate
            
        Returns:
            List of speech segments with start/end times
        """
        if not self.enable_vad or self.vad_model is None:
            # Return full audio as one segment if VAD disabled
            return [{"start": 0.0, "end": len(audio) / sr}]
        
        try:
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).float()
            
            # Get speech timestamps
            speech_timestamps = get_speech_timestamps(
                audio_tensor, 
                self.vad_model,
                sampling_rate=sr,
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=100
            )
            
            # Convert to seconds
            segments = []
            for segment in speech_timestamps:
                segments.append({
                    "start": segment["start"] / sr,
                    "end": segment["end"] / sr
                })
            
            return segments if segments else [{"start": 0.0, "end": len(audio) / sr}]
            
        except Exception as e:
            logger.warning(f"VAD detection failed: {e}")
            return [{"start": 0.0, "end": len(audio) / sr}]
    
    def extract_voice_with_demucs(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract voice from mixed audio using Demucs source separation
        
        Args:
            audio: Input audio array
            sr: Sample rate
            
        Returns:
            Voice-separated audio array
        """
        if not self.enable_source_separation or self.demucs_model is None:
            return audio
        
        try:
            # Convert mono to stereo for Demucs (duplicate channel)
            if audio.ndim == 1:
                audio_stereo = np.stack([audio, audio], axis=0)  # (2, samples)
            else:
                audio_stereo = audio
            
            # Convert to torch tensor and add batch dimension
            audio_tensor = torch.from_numpy(audio_stereo).float()
            if audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(0)  # (1, 2, samples)
            
            logger.info(f"Demucs input shape: {audio_tensor.shape}")
            
            # Apply Demucs model
            with torch.no_grad():
                sources = apply_model(self.demucs_model, audio_tensor)
            
            logger.info(f"Demucs output shape: {sources.shape}")
            
            # Extract vocals (typically index 3 for htdemucs)
            # htdemucs outputs: [drums, bass, other, vocals]
            vocals_stereo = sources[0, 3].numpy()  # Get vocals track (2, samples)
            
            # Convert back to mono (average channels)
            if vocals_stereo.ndim == 2:
                vocals = np.mean(vocals_stereo, axis=0)
            else:
                vocals = vocals_stereo
            
            logger.info("Voice extraction with Demucs completed successfully")
            return vocals
            
        except Exception as e:
            logger.warning(f"Source separation failed: {e}")
            return audio
    
    def intelligent_segmentation(self, audio: np.ndarray, sr: int, 
                               speech_segments: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Segment audio intelligently using VAD boundaries
        
        Args:
            audio: Input audio array
            sr: Sample rate
            speech_segments: List of speech segments from VAD
            
        Returns:
            List of audio segments with metadata
        """
        segments = []
        current_segment_audio = []
        current_segment_start = 0.0
        current_duration = 0.0
        
        for speech_segment in speech_segments:
            start_sample = int(speech_segment["start"] * sr)
            end_sample = int(speech_segment["end"] * sr)
            segment_audio = audio[start_sample:end_sample]
            segment_duration = speech_segment["end"] - speech_segment["start"]
            
            # If adding this segment would exceed target length, finalize current segment
            if current_duration + segment_duration > self.segment_length and current_segment_audio:
                # Finalize current segment
                final_audio = np.concatenate(current_segment_audio)
                segments.append({
                    "audio": final_audio,
                    "start_time": current_segment_start,
                    "end_time": current_segment_start + current_duration,
                    "duration": current_duration
                })
                
                # Start new segment
                current_segment_audio = [segment_audio]
                current_segment_start = speech_segment["start"]
                current_duration = segment_duration
            else:
                # Add to current segment
                current_segment_audio.append(segment_audio)
                if not current_segment_audio or len(current_segment_audio) == 1:
                    current_segment_start = speech_segment["start"]
                current_duration += segment_duration
        
        # Finalize last segment
        if current_segment_audio:
            final_audio = np.concatenate(current_segment_audio)
            segments.append({
                "audio": final_audio,
                "start_time": current_segment_start,
                "end_time": current_segment_start + current_duration,
                "duration": current_duration
            })
        
        return segments
    
    def process_audio_file(self, input_path: Path, output_dir: Path) -> List[Path]:
        """
        Process audio file through the complete pipeline
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory to save processed segments
            
        Returns:
            List of paths to processed audio segments
        """
        logger.info(f"Processing audio file: {input_path}")
        
        # Load audio
        audio, sr = librosa.load(str(input_path), sr=None, mono=True)
        logger.info(f"Loaded audio: {len(audio)} samples at {sr}Hz")
        
        # Step 1: Basic audio cleanup
        audio, sr = self.basic_audio_cleanup(audio, sr)
        logger.info(f"Basic cleanup completed: {sr}Hz")
        
        # Step 2: Source separation (if enabled)
        if self.enable_source_separation:
            audio = self.extract_voice_with_demucs(audio, sr)
            logger.info("Voice extraction completed")
        
        # Step 3: VAD-based preprocessing
        speech_segments = self.detect_voice_activity(audio, sr)
        logger.info(f"Detected {len(speech_segments)} speech segments")
        
        # Step 4: Remove non-speech segments and create speech-only audio
        speech_audio_parts = []
        for segment in speech_segments:
            start_sample = int(segment["start"] * sr)
            end_sample = int(segment["end"] * sr)
            speech_audio_parts.append(audio[start_sample:end_sample])
        
        if speech_audio_parts:
            speech_only_audio = np.concatenate(speech_audio_parts)
        else:
            speech_only_audio = audio
        
        logger.info(f"Speech-only audio: {len(speech_only_audio)} samples")
        
        # Step 5: Intelligent segmentation
        segments = self.intelligent_segmentation(speech_only_audio, sr, speech_segments)
        logger.info(f"Created {len(segments)} intelligent segments")
        
        # Step 6: Save processed segments
        output_paths = []
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, segment in enumerate(segments):
            output_path = output_dir / f"processed_segment_{i:03d}.wav"
            sf.write(str(output_path), segment["audio"], sr)
            output_paths.append(output_path)
            
            # Save metadata
            metadata_path = output_dir / f"processed_segment_{i:03d}_metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump({
                    "segment_index": i,
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "duration": segment["duration"],
                    "sample_rate": sr,
                    "processing_config": {
                        "vad_enabled": self.enable_vad,
                        "source_separation_enabled": self.enable_source_separation,
                        "target_sample_rate": self.target_sr
                    }
                }, f, indent=2)
        
        logger.info(f"Saved {len(output_paths)} processed segments to {output_dir}")
        return output_paths


def process_audio_for_whisper(input_path: Path, 
                            output_dir: Optional[Path] = None,
                            **pipeline_kwargs) -> List[Path]:
    """
    Convenience function to process audio for Whisper transcription
    
    Args:
        input_path: Path to input audio file
        output_dir: Output directory (defaults to temp dir)
        **pipeline_kwargs: Additional arguments for WhisperAudioPipeline
        
    Returns:
        List of paths to processed audio segments
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="whisper_audio_"))
    
    pipeline = WhisperAudioPipeline(**pipeline_kwargs)
    return pipeline.process_audio_file(input_path, output_dir)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Process audio for Whisper transcription")
    parser.add_argument("input_file", help="Input audio file path")
    parser.add_argument("--output-dir", help="Output directory", default="./processed_audio")
    parser.add_argument("--target-sr", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD")
    parser.add_argument("--enable-separation", action="store_true", help="Enable source separation")
    parser.add_argument("--segment-length", type=float, default=30.0, help="Target segment length in seconds")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Process audio
    output_paths = process_audio_for_whisper(
        Path(args.input_file),
        Path(args.output_dir),
        target_sr=args.target_sr,
        enable_vad=not args.no_vad,
        enable_source_separation=args.enable_separation,
        segment_length=args.segment_length
    )
    
    print(f"Processed {len(output_paths)} segments:")
    for path in output_paths:
        print(f"  {path}")