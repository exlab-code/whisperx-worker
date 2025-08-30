#!/usr/bin/env python3
"""
Decoupled Diarization Pipeline - Based on noScribe approach

Implements the intelligent temporal overlap matching exactly as described
in the noScribe project for superior speaker assignment accuracy.
"""

import logging
import torch
import whisperx
import time
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def overlap_len(ts_start: int, ts_end: int, ss_start: int, ss_end: int) -> float:
    """
    Calculate what percentage of the transcription segment is covered by the speaker segment.
    
    This is the core mathematical function from noScribe that determines overlap percentage
    based on the transcription segment duration (not the overlap duration).
    
    Args:
        ts_start: Transcription segment start time (milliseconds)
        ts_end: Transcription segment end time (milliseconds) 
        ss_start: Speaker segment start time (milliseconds)
        ss_end: Speaker segment end time (milliseconds)
        
    Returns:
        Overlap percentage (0.0 to 1.0) - what fraction of transcription is covered
    """
    # Check for non-overlap cases
    if ts_end <= ss_start or ts_start >= ss_end:
        return 0.0
    
    # Calculate transcription segment duration (denominator for percentage)
    ts_len = ts_end - ts_start
    if ts_len <= 0:
        return 0.0
    
    # Find the overlap window
    overlap_start = max(ss_start, ts_start)
    overlap_end = min(ss_end, ts_end)
    
    # Calculate overlap duration
    ol_len = overlap_end - overlap_start
    
    # Return percentage of transcription segment that is covered
    return ol_len / ts_len

def find_speaker(diarization_segments: List[Dict], ts_start_ms: int, ts_end_ms: int) -> str:
    """
    Find the best matching speaker for a transcription segment using noScribe's logic.
    
    This implements the exact decision-making process from noScribe:
    1. Find speaker with highest overlap percentage
    2. Only assign if >= 80% confidence threshold  
    3. Tie-breaker: prefer shorter speaker segments for precision
    
    Args:
        diarization_segments: List of speaker segments from PyAnnote
        ts_start_ms: Transcription segment start time (milliseconds)
        ts_end_ms: Transcription segment end time (milliseconds)
        
    Returns:
        Speaker label or empty string if no confident match
    """
    spkr = ''
    overlap_found = 0.0
    overlap_threshold = 0.8  # 80% confidence threshold
    best_segment_duration = float('inf')  # For tie-breaker rule
    
    logger.debug(f"Finding speaker for segment {ts_start_ms}-{ts_end_ms}ms")
    
    for segment in diarization_segments:
        ss_start_ms = int(segment['start'] * 1000)  # Convert to milliseconds
        ss_end_ms = int(segment['end'] * 1000)
        speaker_label = segment['speaker']
        segment_duration = ss_end_ms - ss_start_ms
        
        # Calculate overlap percentage
        overlap_percentage = overlap_len(ts_start_ms, ts_end_ms, ss_start_ms, ss_end_ms)
        
        logger.debug(f"  vs {speaker_label} ({ss_start_ms}-{ss_end_ms}ms): {overlap_percentage:.3f} overlap")
        
        # Decision logic from noScribe
        should_update = False
        
        if overlap_percentage > overlap_found:
            # This is a better overlap than previous best
            should_update = True
        elif overlap_percentage == overlap_found and overlap_percentage >= overlap_threshold:
            # Tie-breaker rule: prefer shorter speaker segments for precision
            if segment_duration < best_segment_duration:
                should_update = True
                logger.debug(f"    Tie-breaker: shorter segment ({segment_duration}ms vs {best_segment_duration}ms)")
        
        if should_update:
            spkr = speaker_label
            overlap_found = overlap_percentage
            best_segment_duration = segment_duration
            logger.debug(f"    New best: {speaker_label} with {overlap_percentage:.3f} overlap")
    
    # Only return speaker if above confidence threshold
    if overlap_found >= overlap_threshold:
        logger.debug(f"  → Confident assignment: {spkr} ({overlap_found:.3f} >= {overlap_threshold})")
        return spkr
    else:
        logger.debug(f"  → No confident assignment ({overlap_found:.3f} < {overlap_threshold})")
        return ''

def run_decoupled_diarization(audio_path: str,
                             language: Optional[str] = None,
                             batch_size: int = 64,
                             temperature: float = 0,
                             initial_prompt: Optional[str] = None,
                             vad_onset: float = 0.520,
                             vad_offset: float = 0.320,
                             huggingface_token: Optional[str] = None,
                             min_speakers: Optional[int] = None,
                             max_speakers: Optional[int] = None,
                             align_output: bool = True,
                             debug: bool = False) -> Dict[str, Any]:
    """
    Run the complete decoupled pipeline following noScribe's approach:
    1. Stage 1: Pure diarization (who spoke when)
    2. Stage 2: Pure transcription (what was said) 
    3. Stage 3: Intelligent temporal overlap matching
    
    Args:
        audio_path: Path to audio file
        language: Language code or None for auto-detection
        batch_size: Batch size for transcription
        temperature: Sampling temperature
        initial_prompt: Initial prompt for transcription
        vad_onset: VAD onset threshold
        vad_offset: VAD offset threshold
        huggingface_token: HuggingFace access token
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
        align_output: Whether to run word-level alignment
        debug: Whether to print debug information
        
    Returns:
        Complete result with intelligently matched segments
    """
    total_start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16"
    whisper_arch = "./models/faster-whisper-large-v3"
    
    logger.info("🚀 Starting decoupled diarization pipeline (noScribe approach)...")
    
    # Stage 1: Pure Diarization - who spoke when
    logger.info("🎭 Stage 1: Pure diarization...")
    stage1_start = time.time()
    
    diarization_model = whisperx.DiarizationPipeline(
        model_name='pyannote/speaker-diarization@2.1',
        use_auth_token=huggingface_token,
        device=device
    )
    
    audio = whisperx.load_audio(audio_path)
    diarization_result = diarization_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    
    # Convert diarization to list format for processing
    speaker_segments = []
    for segment, _, speaker in diarization_result.itertracks(yield_label=True):
        speaker_segments.append({
            'start': segment.start,
            'end': segment.end, 
            'speaker': speaker
        })
    
    stage1_time = time.time() - stage1_start
    speakers_found = sorted(set(seg['speaker'] for seg in speaker_segments))
    logger.info(f"🎭 Diarization completed: {len(speaker_segments)} segments, {len(speakers_found)} speakers in {stage1_time:.2f}s")
    logger.info(f"   Speakers: {speakers_found}")
    
    # Clean up diarization model
    del diarization_model
    torch.cuda.empty_cache()
    
    # Stage 2: Pure Transcription - what was said
    logger.info("📝 Stage 2: Pure transcription...")
    stage2_start = time.time()
    
    # Load transcription model
    asr_options = {
        "temperatures": [temperature],
        "initial_prompt": initial_prompt
    }
    
    vad_options = {
        "vad_onset": vad_onset,
        "vad_offset": vad_offset
    }
    
    model = whisperx.load_model(
        whisper_arch, device,
        compute_type=compute_type,
        language=language,
        asr_options=asr_options,
        vad_options=vad_options
    )
    
    # Run transcription
    result = model.transcribe(audio, batch_size=batch_size)
    detected_language = result["language"]
    
    stage2_time = time.time() - stage2_start
    logger.info(f"📝 Transcription completed: {len(result['segments'])} segments in {stage2_time:.2f}s")
    logger.info(f"   Language: {detected_language}")
    
    # Clean up transcription model
    del model
    torch.cuda.empty_cache()
    
    # Stage 2.5: Word-level Alignment (optional)
    if align_output and detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_TORCH | whisperx.alignment.DEFAULT_ALIGN_MODELS_HF:
        logger.info("🎯 Stage 2.5: Word-level alignment...")
        align_start = time.time()
        
        model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
        # Clean up alignment model
        del model_a
        torch.cuda.empty_cache()
        
        align_time = time.time() - align_start
        logger.info(f"🎯 Alignment completed in {align_time:.2f}s")
    
    # Stage 3: Intelligent Temporal Matching (the noScribe magic)
    logger.info("🧠 Stage 3: Intelligent temporal matching...")
    stage3_start = time.time()
    
    matched_segments = []
    assignment_stats = {"confident": 0, "uncertain": 0}
    
    for segment in result["segments"]:
        # Convert to milliseconds as in noScribe
        start_ms = round(segment.get('start', 0) * 1000)
        end_ms = round(segment.get('end', 0) * 1000)
        
        # Find best speaker using noScribe's logic
        assigned_speaker = find_speaker(speaker_segments, start_ms, end_ms)
        
        # Create final segment
        final_segment = {
            "start": float(segment.get('start', 0)),
            "end": float(segment.get('end', 0)),
            "text": segment.get('text', '').strip(),
            "speaker": assigned_speaker if assigned_speaker else "SPEAKER_UNKNOWN",
            "confidence": float(segment.get('avg_logprob', 0.0))
        }
        
        # Add word-level data if available
        if 'words' in segment:
            final_segment["words"] = segment['words']
        
        matched_segments.append(final_segment)
        
        # Track assignment quality
        if assigned_speaker:
            assignment_stats["confident"] += 1
        else:
            assignment_stats["uncertain"] += 1
    
    stage3_time = time.time() - stage3_start
    total_time = time.time() - total_start_time
    
    # Calculate assignment rate
    total_assignments = assignment_stats["confident"] + assignment_stats["uncertain"]
    assignment_rate = (assignment_stats["confident"] / total_assignments * 100) if total_assignments > 0 else 0
    
    logger.info(f"🧠 Matching completed in {stage3_time:.2f}s")
    logger.info(f"   Confident assignments: {assignment_stats['confident']}")
    logger.info(f"   Uncertain assignments: {assignment_stats['uncertain']}")
    logger.info(f"   Assignment rate: {assignment_rate:.1f}%")
    logger.info(f"✅ Total pipeline time: {total_time:.2f}s")
    
    # Compile final result
    final_result = {
        "segments": matched_segments,
        "detected_language": detected_language,
        "processing_info": {
            "pipeline_type": "decoupled_noscribe",
            "total_time": total_time,
            "stage1_diarization_time": stage1_time,
            "stage2_transcription_time": stage2_time,
            "stage3_matching_time": stage3_time,
            "speakers_detected": len(speakers_found),
            "transcription_segments": len(result["segments"]),
            "final_segments": len(matched_segments),
            "confident_assignments": assignment_stats["confident"],
            "assignment_rate": assignment_rate,
            "overlap_threshold": 0.8,
            "alignment_enabled": align_output
        }
    }
    
    return final_result