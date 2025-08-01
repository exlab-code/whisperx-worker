import os
import shutil
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import download_files_from_urls, rp_cleanup
from rp_schema import INPUT_VALIDATIONS
from predict import Predictor, Output

MODEL = Predictor()
MODEL.setup()

def cleanup_job_files(job_id, jobs_directory='/jobs'):
    job_path = os.path.join(jobs_directory, job_id)
    if os.path.exists(job_path):
        try:
            shutil.rmtree(job_path)
            print(f"Removed job directory: {job_path}")
        except Exception as e:
            print(f"Error removing job directory {job_path}: {str(e)}")
    else:
        print(f"Job directory not found: {job_path}")

def run(job):
    from datetime import datetime
    import base64
    import tempfile
    
    processing_start_time = datetime.now()
    job_input = job['input']
    job_id = job['id']
    
    # Extract session info for logging
    session_id = job_input.get("session_id", "unknown")
    chunk_index = job_input.get("chunk_index", 0)
    filename = job_input.get("filename", "audio.wav")
    
    print(f"🎙️ Processing [{session_id}:{chunk_index}] via WhisperX GPU serverless")
    
    # Input validation
    validated_input = validate(job_input, INPUT_VALIDATIONS)
    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    
    # Decode base64 audio data
    try:
        audio_b64 = job_input["audio_b64"]
        audio_data = base64.b64decode(audio_b64)
        print(f"📦 Decoded audio: {len(audio_data)} bytes")
        
        # Create temporary file for transcription
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_data)
            audio_file_path = temp_file.name
            
        print(f"Audio saved to temporary file → {audio_file_path}")
    except Exception as e:
        print(f"❌ Audio base64 decoding failed: {e}")
        return {"error": f"audio decoding: {e}"}
    
    # Prepare input for prediction
    predict_input = {
        'audio_file': audio_file_path,
        'language': job_input.get('language'),
        'language_detection_min_prob': job_input.get('language_detection_min_prob', 0),
        'language_detection_max_tries': job_input.get('language_detection_max_tries', 5),
        'initial_prompt': job_input.get('initial_prompt'),
        'batch_size': job_input.get('batch_size', 64),
        'temperature': job_input.get('temperature', 0),
        'vad_onset': job_input.get('vad_onset', 0.300),  # More sensitive
        'vad_offset': job_input.get('vad_offset', 0.200),  # More sensitive
        'align_output': job_input.get('align_output', True),  # Enable word-level alignment
        'diarization': job_input.get('diarization', True),  # Enable speaker diarization
        'huggingface_access_token': job_input.get('huggingface_access_token'),
        'min_speakers': job_input.get('min_speakers'),
        'max_speakers': job_input.get('max_speakers'),
        'debug': job_input.get('debug', False)
    }
    
    # Run prediction
    try:
        result = MODEL.predict(**predict_input)
        
        # Processing completion time
        processing_end_time = datetime.now()
        processing_duration = (processing_end_time - processing_start_time).total_seconds()
        
        # Format output to match Railway app expectations
        formatted_segments = []
        for segment in result.segments:
            formatted_segment = {
                "start": segment.get("start", 0.0),
                "end": segment.get("end", 0.0),
                "text": segment.get("text", ""),
                "speaker": segment.get("speaker", "SPEAKER_00")
            }
            formatted_segments.append(formatted_segment)
        
        # Build response matching Railway app format
        response = {
            "text": " ".join([seg["text"] for seg in formatted_segments]),
            "language": result.detected_language,
            "segments": formatted_segments,
            "processing_info": {
                "session_id": session_id,
                "chunk_index": chunk_index,
                "filename": filename,
                "duration_seconds": processing_duration,
                "audio_bytes": len(audio_data),
                "segments_count": len(formatted_segments)
            }
        }
        
        print(f"✅ Completed [{session_id}:{chunk_index}] in {processing_duration:.2f}s - {len(formatted_segments)} segments")
        
        # Cleanup temporary file and job files
        try:
            os.unlink(audio_file_path)
            print(f"🗑️ Cleaned up temporary file: {audio_file_path}")
        except Exception as cleanup_error:
            print(f"⚠️ Failed to cleanup temp file: {cleanup_error}")
        
        rp_cleanup.clean(['input_objects'])
        cleanup_job_files(job_id)
        
        return response
    except Exception as e:
        # Cleanup temporary file on error
        try:
            os.unlink(audio_file_path)
        except:
            pass
        print(f"❌ Prediction failed for [{session_id}:{chunk_index}]: {e}")
        return {"error": str(e)}

runpod.serverless.start({"handler": run})
