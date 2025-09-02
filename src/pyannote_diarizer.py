"""
Enhanced diarization using pyannote.audio 3.1 pipeline
Replaces whisperx's built-in diarization with state-of-the-art models
"""

import torch
import gc
import numpy as np
import pandas as pd
from pyannote.audio import Pipeline
from typing import Optional, Union
import logging

# Global pipeline cache to avoid reloading models
pipeline = None

def initialize_pipeline(auth_token: str, device: str = "cuda") -> None:
    """
    Initialize the pyannote.audio 3.1 pipeline once and cache it.
    
    Args:
        auth_token: HuggingFace access token
        device: Device to run on ('cuda' or 'cpu')
    """
    global pipeline
    if pipeline is not None:
        return
    
    try:
        print("Initializing pyannote.audio 3.1 pipeline...")
        
        # Use the latest pyannote/speaker-diarization-3.1 pipeline
        # This removes ONNX runtime issues and runs pure PyTorch
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token
        )
        
        pipeline.to(torch.device(device))
        print(f"Pyannote 3.1 pipeline initialized successfully on {device}")
        
    except Exception as e:
        print(f"Error initializing pyannote pipeline: {e}")
        raise e

def run_diarization(
    audio_tensor: np.ndarray, 
    min_speakers: Optional[int] = None, 
    max_speakers: Optional[int] = None, 
    auth_token: str = None,
    device: str = "cuda"
) -> dict:
    """
    Run speaker diarization using pyannote.audio 3.1 pipeline.
    
    Args:
        audio_tensor: Audio data as numpy array (mono, 16kHz)
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers  
        auth_token: HuggingFace access token
        device: Device to run on
        
    Returns:
        Diarization segments compatible with whisperx.assign_word_speakers()
    """
    if auth_token is None:
        raise ValueError("HuggingFace access token is required for pyannote models")
    
    # Initialize pipeline if not already loaded
    initialize_pipeline(auth_token, device)
    
    try:
        # Convert numpy array to torch tensor
        if isinstance(audio_tensor, np.ndarray):
            audio_data = torch.from_numpy(audio_tensor).float()
        else:
            audio_data = audio_tensor.float()
            
        # Ensure audio is in correct shape [1, samples] for mono audio
        if audio_data.ndim == 1:
            audio_data = audio_data.unsqueeze(0)
        elif audio_data.ndim == 2 and audio_data.shape[0] > audio_data.shape[1]:
            # If shape is [samples, 1], transpose to [1, samples]
            audio_data = audio_data.T
            
        # Create input dict for pyannote pipeline
        waveform_input = {
            "waveform": audio_data, 
            "sample_rate": 16000
        }
        
        # Run diarization with optional speaker constraints
        diarization_kwargs = {}
        if min_speakers is not None:
            diarization_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarization_kwargs["max_speakers"] = max_speakers
            
        print(f"Running diarization with constraints: min_speakers={min_speakers}, max_speakers={max_speakers}")
        diarization = pipeline(waveform_input, **diarization_kwargs)
        
        # Convert pyannote Annotation to pandas DataFrame format expected by whisperx.assign_word_speakers
        segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": segment.start,
                "end": segment.end, 
                "speaker": speaker
            })
        
        # Convert to DataFrame - this is what whisperx.assign_word_speakers expects
        diarization_df = pd.DataFrame(segments)
        
        print(f"Diarization complete: found {len(segments)} segments from {len(set(s['speaker'] for s in segments))} speakers")
        print(f"DataFrame format: {diarization_df.columns.tolist()}")
        
        # Return DataFrame format expected by whisperx.assign_word_speakers  
        return diarization_df
        
    except Exception as e:
        print(f"Error during diarization: {e}")
        raise e
    finally:
        # Clean up GPU memory
        gc.collect()
        torch.cuda.empty_cache()

def get_pipeline_info() -> dict:
    """Return information about the loaded pipeline"""
    global pipeline
    if pipeline is None:
        return {"status": "not_initialized"}
    
    return {
        "status": "initialized",
        "pipeline_name": "pyannote/speaker-diarization-3.1",
        "device": str(pipeline.device) if hasattr(pipeline, 'device') else "unknown"
    }