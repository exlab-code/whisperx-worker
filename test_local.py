#!/usr/bin/env python3
"""
Local test script for the modified WhisperX handler
Tests base64 input and audio preprocessing integration
"""

import json
import sys
import os

# Add src to path so we can import the handler
sys.path.append('src')

def test_handler():
    """Test the modified WhisperX handler with base64 input"""
    print("🧪 Testing WhisperX handler with base64 input...")
    
    try:
        # Import the handler
        from rp_handler import run
        
        # Load test input
        with open('test_input_base64.json', 'r') as f:
            test_job = json.load(f)
        
        # Add job ID as required by handler
        test_job['id'] = 'test_job_001'
        
        print(f"Input: {json.dumps(test_job, indent=2)}")
        print("=" * 50)
        
        # Run the handler
        result = run(test_job)
        
        print("✅ Handler completed successfully!")
        print("Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Validate expected fields
        expected_fields = ['text', 'language', 'segments', 'processing_info']
        missing_fields = [field for field in expected_fields if field not in result]
        
        if missing_fields:
            print(f"⚠️ Missing expected fields: {missing_fields}")
        else:
            print("✅ All expected fields present")
            
        # Check processing info
        if 'processing_info' in result:
            proc_info = result['processing_info']
            print(f"📊 Processing metrics:")
            print(f"   - Model: {proc_info.get('model', 'unknown')}")
            print(f"   - RTF: {proc_info.get('real_time_factor', 'N/A')}")
            print(f"   - Speakers: {proc_info.get('speakers_detected', 'N/A')}")
            print(f"   - Segments: {proc_info.get('segments_count', 'N/A')}")
            print(f"   - Diarization: {proc_info.get('diarization', 'N/A')}")
            print(f"   - Audio preprocessing: {proc_info.get('audio_preprocessing', 'N/A')}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the WhisperX worker directory")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_handler()
    sys.exit(0 if success else 1)