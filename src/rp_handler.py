import os
import shutil
import runpod
from pathlib import Path
from predict import Predictor

# Initialize predictor with lazy loading for health check compatibility
MODEL = Predictor()

def run(job):
    """Clean noScribe-based RunPod handler"""
    
    # Lazy initialization: Setup models on first request to avoid health check timeout  
    if MODEL.faster_whisper_model is None:
        print("🚀 First request - loading models (lazy initialization)...")
        MODEL.setup()
        print("✅ Models loaded successfully, ready for predictions!")
    
    job_input = job['input']
    job_id = job['id']
    
    try:
        # Basic input validation
        if 'audio' not in job_input and 'audio_base64' not in job_input:
            return {"error": "Either 'audio' URL or 'audio_base64' is required"}
        
        # Download audio file if URL provided
        if 'audio' in job_input:
            from runpod.serverless.utils import download_files_from_urls
            audio_file_path = download_files_from_urls(job_id, [job_input['audio']])[0]
        else:
            # Handle base64 audio (simplified for now)
            return {"error": "Base64 audio not yet implemented"}
        
        # Map RunPod inputs to our clean predict interface
        result = MODEL.predict(
            audio_file=Path(audio_file_path),
            language=job_input.get('language'),
            num_speakers=job_input.get('num_speakers'),
            enable_diarization=job_input.get('enable_diarization', True),
            speech_pad_ms=job_input.get('speech_pad_ms', 400)
        )
        
        # Cleanup job files
        job_path = f"/jobs/{job_id}"
        if os.path.exists(job_path):
            shutil.rmtree(job_path)
            print(f"Cleaned up job directory: {job_path}")
        
        return {
            "segments": result.segments,
            "language": result.language
        }
        
    except Exception as e:
        print(f"❌ Error in handler: {e}")
        return {"error": str(e)}

runpod.serverless.start({"handler": run})
