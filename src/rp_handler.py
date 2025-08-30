# at top of rp_handler.py (or speaker_processing.py)
from dotenv import load_dotenv, find_dotenv
import os

# find and load your .env file
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN")# 


######SETTING HF_TOKENT#############

from speaker_profiles import load_embeddings, relabel  # top of file
from speaker_processing import process_diarized_output,enroll_profiles, identify_speakers_on_segments, load_known_speakers_from_samples, identify_speaker, relabel_speakers_by_avg_similarity
import logging
from huggingface_hub import login, whoami
import torch
import numpy as np
from dotenv import load_dotenv, find_dotenv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from speechbrain.pretrained import EncoderClassifier # type: ignore

def spk_embed(wave_16k_mono: np.ndarray) -> np.ndarray:
    wav = torch.tensor(wave_16k_mono).unsqueeze(0).to(device)
    return ecapa.encode_batch(wav).squeeze(0).cpu().numpy()

def to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Grab the HF_TOKEN from environment
raw_token = os.environ.get("HF_TOKEN", "")
hf_token = raw_token.strip()

if not hf_token.startswith("hf_"):
    print(f"Token malformed or missing 'hf_' prefix. Forcing correction...")
    hf_token = "h" + hf_token  # Force adding the 'h' (temporary fix)

#print(f" Final HF_TOKEN used: #{hf_token}")
if hf_token:
    try:
        logger.debug(f"HF_TOKEN Loaded: {repr(hf_token[:10])}...")  # Show only start of token for security
        login(token=hf_token, add_to_git_credential=False)  # Safe for container runs
        user = whoami(token=hf_token)
        logger.info(f"Hugging Face Authenticated as: {user['name']}")
    except Exception as e:
        logger.error(" Failed to authenticate with Hugging Face", exc_info=True)
else:
    logger.warning("No Hugging Face token found in HF_TOKEN environment variable.")
##############

import shutil
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import download_files_from_urls, rp_cleanup
from rp_schema import INPUT_VALIDATIONS
from predict import Predictor, Output
import os
import copy
import logging

# Audio preprocessing temporarily disabled for debugging
AUDIO_PIPELINE_AVAILABLE = False
import sys
# Create a custom logger
logger = logging.getLogger("rp_handler")
logger.setLevel(logging.DEBUG)  # capture everything at DEBUG or above

# Create console handler and set level to DEBUG
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)

# Create file handler to write logs to 'container_log.txt'
file_handler = logging.FileHandler("container_log.txt", mode="a")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)

# Add both handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)




MODEL = Predictor()
MODEL.setup()

def cleanup_job_files(job_id, jobs_directory='/jobs'):
    job_path = os.path.join(jobs_directory, job_id)
    if os.path.exists(job_path):
        try:
            shutil.rmtree(job_path)
            logger.info(f"Removed job directory: {job_path}")
        except Exception as e:
            logger.error(f"Error removing job directory {job_path}: {str(e)}", exc_info=True)
    else:
        logger.debug(f"Job directory not found: {job_path}")

# --------------------------------------------------------------------
# main serverless entry-point
# --------------------------------------------------------------------
error_log = []
def run(job):
    from datetime import datetime
    processing_start_time = datetime.now()
    
    job_id     = job["id"]
    job_input  = job["input"]

    # ------------- validate basic schema ----------------------------
    validated = validate(job_input, INPUT_VALIDATIONS)
    if "errors" in validated:
        return {"error": validated["errors"]}

    # ------------- 1) download audio file from URL ------------------------
    try:
        from audio_downloader import download_audio_file, cleanup_temp_file
        
        logger.info(f"🎙️ Processing audio file via WhisperX GPU serverless")
        
        # Download audio file from URL
        audio_url = job_input["audio_url"]
        audio_file_path, download_info = download_audio_file(audio_url)
        
        logger.info(f"📥 Downloaded audio: {download_info['bytes_downloaded']:,} bytes")
        logger.debug(f"File type: {download_info['file_extension']}, Content-Type: {download_info['content_type']}")
        logger.debug(f"Audio saved to temporary file → {audio_file_path}")
        
    except Exception as e:
        logger.error("Audio download failed", exc_info=True)
        return {"error": f"audio download: {e}"}

    # ------------- 2) download speaker profiles (optional) ----------
    speaker_profiles = job_input.get("speaker_samples", [])
    embeddings = {}
    if speaker_profiles:
        try:
            embeddings = load_known_speakers_from_samples(
                speaker_profiles,
                huggingface_access_token=hf_token  # or job_input.get("huggingface_access_token")
            )
            logger.info(f"Enrolled {len(embeddings)} speaker profiles successfully.")
        except Exception as e:
            logger.error("Enrollment failed", exc_info=True)
            output_dict["warning"] = f"Enrollment skipped: {e}"
        # urls = [s.get("url") for s in speaker_profiles if s.get("url")]
        # if urls:
        #     try:
        #         local_paths = download_files_from_urls(job_id, urls)
        #         for s, path in zip(speaker_profiles, local_paths):
        #             s["file_path"] = path  # mutate in-place
        #             logger.debug(f"Profile {s.get('name')} → {path}")

        #         # Now enroll profiles using the updated speaker_profiles with local file paths
        #         embeddings = enroll_profiles(speaker_profiles)
        #         logger.info(f"Enrolled {len(embeddings)} speaker profiles successfully.")
        #     except Exception as e:
        #         logger.error("Enrollment failed", exc_info=True)
        #         output_dict["warning"] = f"Enrollment skipped: {e}"
    # ----------------------------------------------------------------

    # ------------- 3) Choose pipeline: coupled or decoupled -------------
    use_decoupled = job_input.get("decoupled_diarization", True)  # Default to new approach
    
    predict_input = {
        "audio_file"               : audio_file_path,
        "language"                 : job_input.get("language"),
        "language_detection_min_prob": job_input.get("language_detection_min_prob", 0),
        "language_detection_max_tries": job_input.get("language_detection_max_tries", 5),
        "initial_prompt"           : job_input.get("initial_prompt"),
        "batch_size"               : job_input.get("batch_size", 64),
        "temperature"              : job_input.get("temperature", 0),
        # Conservative VAD parameters to prevent cutting off speech beginnings/endings
        # Higher values = less aggressive VAD = more speech preserved
        "vad_onset"                : job_input.get("vad_onset", 0.520),   # Less aggressive than default 0.500
        "vad_offset"               : job_input.get("vad_offset", 0.320),  # Less aggressive than default 0.363
        "align_output"             : job_input.get("align_output", True),   # Enable word-level alignment
        "diarization"              : job_input.get("diarization", True),  # Enable speaker diarization
        "huggingface_access_token" : job_input.get("huggingface_access_token") or hf_token,
        "min_speakers"             : job_input.get("min_speakers"),
        "max_speakers"             : job_input.get("max_speakers"),
        "debug"                    : job_input.get("debug", False),
    }

    # ------------- Process audio with chosen pipeline -------------

    try:
        transcription_start_time = datetime.now()
        
        if use_decoupled:
            # Use new decoupled pipeline (noScribe approach)
            logger.info(f"🔄 Using decoupled diarization pipeline")
            from decoupled_pipeline import run_decoupled_diarization
            
            result = run_decoupled_diarization(
                audio_path=audio_file_path,
                language=predict_input.get("language"),
                batch_size=predict_input.get("batch_size", 64),
                temperature=predict_input.get("temperature", 0),
                initial_prompt=predict_input.get("initial_prompt"),
                vad_onset=predict_input.get("vad_onset", 0.520),
                vad_offset=predict_input.get("vad_offset", 0.320),
                huggingface_token=predict_input.get("huggingface_access_token"),
                min_speakers=predict_input.get("min_speakers"),
                max_speakers=predict_input.get("max_speakers"),
                align_output=predict_input.get("align_output", True),
                debug=predict_input.get("debug", False)
            )
            
            # Result is already in the correct format for decoupled pipeline
            segments = result["segments"]
            detected_language = result["detected_language"]
            
        else:
            # Use original coupled WhisperX pipeline
            logger.info(f"🔄 Using coupled WhisperX pipeline")
            logger.debug(f"Starting transcription with VAD onset={predict_input['vad_onset']}, offset={predict_input['vad_offset']}")
            
            result = MODEL.predict(**predict_input)             # <-- heavy job
            
            # Log transcription results for debugging
            if hasattr(result, 'segments') and result.segments:
                first_seg = result.segments[0] if result.segments else None
                last_seg = result.segments[-1] if result.segments else None
                logger.debug(f"Transcription produced {len(result.segments)} segments")
                if first_seg:
                    logger.debug(f"First segment: {first_seg.get('start', 0):.3f}s - {first_seg.get('end', 0):.3f}s")
                if last_seg and first_seg != last_seg:
                    logger.debug(f"Last segment: {last_seg.get('start', 0):.3f}s - {last_seg.get('end', 0):.3f}s")
            else:
                logger.warning("No segments produced by transcription")
                
            segments = result.segments
            detected_language = getattr(result, 'detected_language', 'en')
            
    except Exception as e:
        logger.error("Transcription pipeline failed", exc_info=True)
        return {"error": f"prediction: {e}"}

    # Convert output to Railway app format
    from datetime import datetime
    import time
    transcription_end_time = datetime.now()
    
    if use_decoupled:
        # Decoupled pipeline already returns segments in correct format
        cleaned_segments = segments
        full_text_parts = [seg["text"] for seg in segments if seg.get("text", "").strip()]
        
        # Add processing info from decoupled pipeline
        decoupled_processing_info = result.get("processing_info", {})
        
    else:
        # Process segments from coupled WhisperX pipeline
        cleaned_segments = []
        full_text_parts = []
        
        for segment in segments:
            # Extract text and basic info
            segment_text = segment.get('text', '').strip()
            if not segment_text:
                continue
                
            # Extract speaker info (WhisperX provides real diarization)
            speaker = segment.get('speaker', 'SPEAKER_00')
            if not speaker or speaker in ['None', None]:
                speaker = 'SPEAKER_00'
                
            cleaned_segments.append({
                "start": float(segment.get('start', 0)),
                "end": float(segment.get('end', 0)),
                "text": segment_text,
                "speaker": speaker,
                "confidence": float(segment.get('avg_logprob', 0.0))
            })
            
            full_text_parts.append(segment_text)
    
    # Calculate metrics - get actual audio duration from file
    full_text = " ".join(full_text_parts)
    try:
        # Get actual audio duration from the temporary file
        import librosa
        actual_audio, sr = librosa.load(audio_file_path, sr=None)
        audio_duration = len(actual_audio) / sr
        logger.debug(f"Actual audio duration: {audio_duration:.3f}s from file analysis")
    except Exception as e:
        # Fallback to segment-based calculation if file analysis fails
        audio_duration = max([seg['end'] for seg in cleaned_segments]) if cleaned_segments else 30.0
        logger.warning(f"Failed to get actual audio duration, using segments: {audio_duration:.3f}s - {e}")
        
    # Validate segment timestamps against actual duration
    if cleaned_segments and audio_duration > 0:
        max_segment_time = max([seg['end'] for seg in cleaned_segments])
        if max_segment_time > audio_duration * 1.1:  # 10% tolerance
            logger.warning(f"Segment timestamps ({max_segment_time:.3f}s) exceed audio duration ({audio_duration:.3f}s) - possible VAD issue")
    transcription_time = (transcription_end_time - transcription_start_time).total_seconds() if 'transcription_start_time' in locals() else 0
    total_processing_time = (transcription_end_time - processing_start_time).total_seconds() if 'processing_start_time' in locals() else 0
    rtf = total_processing_time / audio_duration if audio_duration > 0 else 0
    
    # Create Railway app compatible output
    base_processing_info = {
        "transcription_time": transcription_time,
        "total_processing_time": total_processing_time,
        "real_time_factor": rtf,
        "model": "whisperx-large-v3",
        "compute_type": "float16",
        "device": "cuda",
        "speakers_detected": len(set(seg["speaker"] for seg in cleaned_segments)),
        "segments_count": len(cleaned_segments),
        "serverless": True,
        "diarization": True,
        "audio_preprocessing": AUDIO_PIPELINE_AVAILABLE,
        "pipeline_type": "decoupled_noscribe" if use_decoupled else "coupled_whisperx"
    }
    
    # Add decoupled-specific processing info if available
    if use_decoupled and 'decoupled_processing_info' in locals():
        base_processing_info.update(decoupled_processing_info)
    
    output_dict = {
        "text": full_text,
        "language": detected_language,
        "language_probability": 0.9,  # WhisperX doesn't provide this directly
        "duration": float(audio_duration),
        "segments": cleaned_segments,
        "processing_info": base_processing_info
    }
    
    # Log success message
    logger.info(f"✅ WhisperX RTF: {rtf:.2f} | {len(cleaned_segments)} segments | {len(set(seg['speaker'] for seg in cleaned_segments))} speakers")
    # ------------------------------------------------embedding-info----------------
    # 4) speaker verification (optional)
    if embeddings:
        try:
            segments_with_speakers = identify_speakers_on_segments(
                segments=output_dict["segments"],
                audio_path=audio_file_path,
                enrolled=embeddings,
                threshold=0.1  # Adjust threshold as needed
            )
            #output_dict["segments"] = segments_with_speakers
            segments_with_final_labels = relabel_speakers_by_avg_similarity(segments_with_speakers)
            output_dict["segments"] = segments_with_final_labels
            logger.info("Speaker identification completed successfully.")
        except Exception as e:
            logger.error("Speaker identification failed", exc_info=True)
            output_dict["warning"] = f"Speaker identification skipped: {e}"
    else:
        logger.info("No enrolled embeddings available; skipping speaker identification.")

    # 4-Cleanup and return output_dict normally
    try:
        # Clean up temporary audio file using audio downloader cleanup
        if 'audio_file_path' in locals():
            cleanup_temp_file(audio_file_path)
        
        rp_cleanup.clean(["input_objects"])
        cleanup_job_files(job_id)
    except Exception as e:
        logger.warning(f"Cleanup issue: {e}", exc_info=True)

    return output_dict

runpod.serverless.start({"handler": run})


#     embeddings = {} # ensure the name is always bound
#     if job_input.get("speaker_verification", True):
#         logger.info(f"Speaker-verification requested: True")
#         try:
#             embeddings = load_known_speakers_from_samples(
#                 speaker_profiles,
#                 huggingface_access_token=predict_input["huggingface_access_token"]
#             )
#             logger.info(f"  • Enrolled {len(embeddings)} profiles")
#         except Exception as e:
#             logger.error("Failed loading speaker profiles", exc_info=True)
#             output_dict["warning"] = f"enrollment skipped: {e}"

#         embedding_log_data = None  # Initialize here to avoid UnboundLocalError

#         if embeddings:  # only attempt verification if we actually got something
#             try:
#                 output_dict, embedding_log_data = process_diarized_output(
#                     output_dict,
#                     audio_file_path,
#                     embeddings,
#                     huggingface_access_token=job_input.get("huggingface_access_token"),
#                     return_logs=False # <-- set to True for debugging
#             except Exception as e:
#                 logger.error("Error during speaker verification", exc_info=True)
#                 output_dict["warning"] = f"verification skipped: {e}"
#         else:
#             logger.info("No embeddings to verify against; skipping verification step")

#     if embedding_log_data:
#         output_dict["embedding_logs"] = embedding_log_data

#     # 5) cleanup
#     try:
#         rp_cleanup.clean(["input_objects"])
#         cleanup_job_files(job_id)
#     except Exception as e:
#         logger.warning(f"Cleanup issue: {e}", exc_info=True)

#         # If you have any errors, attach them to the output
#     if error_log:
#         output_dict["error_log"] = "\n".join(error_log)

#     return output_dict

# runpod.serverless.start({"handler": run})