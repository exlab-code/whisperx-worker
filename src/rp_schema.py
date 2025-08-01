INPUT_VALIDATIONS = {
    'audio_b64': {
        'type': str,
        'required': True
    },
    'session_id': {
        'type': str,
        'required': False,
        'default': 'unknown'
    },
    'chunk_index': {
        'type': int,
        'required': False,
        'default': 0
    },
    'filename': {
        'type': str,
        'required': False,
        'default': 'audio.wav'
    },
    'language': {
        'type': str,
        'required': False,
        'default': None
    },
    'language_detection_min_prob': {
        'type': float,
        'required': False,
        'default': 0
    },
    'language_detection_max_tries': {
        'type': int,
        'required': False,
        'default': 5
    },
    'initial_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'batch_size': {
        'type': int,
        'required': False,
        'default': 64
    },
    'temperature': {
        'type': float,
        'required': False,
        'default': 0
    },
    'vad_onset': {
        'type': float,
        'required': False,
        'default': 0.500
    },
    'vad_offset': {
        'type': float,
        'required': False,
        'default': 0.363
    },
    'align_output': {
        'type': bool,
        'required': False,
        'default': False
    },
    'diarization': {
        'type': bool,
        'required': False,
        'default': False
    },
    'huggingface_access_token': {
        'type': str,
        'required': False,
        'default': None
    },
    'min_speakers': {
        'type': int,
        'required': False,
        'default': None
    },
    'max_speakers': {
        'type': int,
        'required': False,
        'default': None
    },
    'debug': {
        'type': bool,
        'required': False,
        'default': False
    }
}