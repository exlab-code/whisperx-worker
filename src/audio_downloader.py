#!/usr/bin/env python3
"""
Audio file downloader for URL-based input processing

Handles downloading audio files from URLs with proper error handling,
timeout protection, and file size limits.
"""

import os
import tempfile
import requests
import logging
from typing import Tuple, Optional
from urllib.parse import urlparse
from pathlib import Path

logger = logging.getLogger(__name__)

# Configuration
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB limit for audio files
DOWNLOAD_TIMEOUT = 120  # 2 minutes timeout
CHUNK_SIZE = 8192  # Download in 8KB chunks
SUPPORTED_EXTENSIONS = {'.webm', '.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac'}

def validate_url(url: str) -> bool:
    """
    Validate that the URL is properly formatted and uses allowed schemes.
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    try:
        parsed = urlparse(url)
        return all([
            parsed.scheme in ('http', 'https'),
            parsed.netloc,
            len(url) < 2048  # Reasonable URL length limit
        ])
    except Exception as e:
        logger.error(f"URL validation error: {e}")
        return False

def get_file_extension(url: str, content_type: Optional[str] = None) -> str:
    """
    Determine appropriate file extension from URL or content type.
    
    Args:
        url: Source URL
        content_type: HTTP Content-Type header
        
    Returns:
        File extension (e.g., '.webm')
    """
    # Try to get extension from URL path
    parsed_url = urlparse(url)
    url_ext = Path(parsed_url.path).suffix.lower()
    
    if url_ext in SUPPORTED_EXTENSIONS:
        return url_ext
    
    # Fall back to content type mapping
    content_type_map = {
        'audio/webm': '.webm',
        'audio/mpeg': '.mp3',
        'audio/mp3': '.mp3', 
        'audio/wav': '.wav',
        'audio/wave': '.wav',
        'audio/x-wav': '.wav',
        'audio/mp4': '.m4a',
        'audio/aac': '.aac',
        'audio/ogg': '.ogg',
        'audio/flac': '.flac'
    }
    
    if content_type:
        # Remove charset and other parameters
        main_type = content_type.split(';')[0].strip().lower()
        return content_type_map.get(main_type, '.webm')  # Default to webm
    
    return '.webm'  # Safe default

def download_audio_file(url: str) -> Tuple[str, dict]:
    """
    Download audio file from URL to temporary file with comprehensive error handling.
    
    Args:
        url: URL to download audio from
        
    Returns:
        Tuple of (temp_file_path, download_info)
        
    Raises:
        ValueError: For invalid URLs or unsupported formats
        requests.RequestException: For network/HTTP errors
        OSError: For file system errors
    """
    # Validate URL
    if not validate_url(url):
        raise ValueError(f"Invalid or unsupported URL: {url}")
    
    logger.info(f"📥 Starting download from: {url}")
    
    download_info = {
        "url": url,
        "bytes_downloaded": 0,
        "content_type": None,
        "file_extension": None,
        "temp_file_path": None
    }
    
    temp_file = None
    
    try:
        # Start download with streaming
        response = requests.get(
            url, 
            stream=True, 
            timeout=DOWNLOAD_TIMEOUT,
            headers={
                'User-Agent': 'WhisperX-Worker/1.0'
            }
        )
        response.raise_for_status()
        
        # Check content type and size
        content_type = response.headers.get('content-type', '')
        content_length = response.headers.get('content-length')
        
        download_info["content_type"] = content_type
        
        # Validate file size if provided
        if content_length:
            try:
                size = int(content_length)
                if size > MAX_FILE_SIZE:
                    raise ValueError(f"File too large: {size} bytes (max: {MAX_FILE_SIZE})")
                logger.debug(f"Expected file size: {size:,} bytes")
            except ValueError as e:
                if "File too large" in str(e):
                    raise
                logger.warning(f"Invalid content-length header: {content_length}")
        
        # Determine file extension
        file_ext = get_file_extension(url, content_type)
        download_info["file_extension"] = file_ext
        
        logger.debug(f"Detected file type: {file_ext} (Content-Type: {content_type})")
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            suffix=file_ext,
            delete=False,
            prefix="whisperx_audio_"
        )
        
        download_info["temp_file_path"] = temp_file.name
        
        # Download with progress tracking
        bytes_downloaded = 0
        
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:  # Filter out keep-alive chunks
                temp_file.write(chunk)
                bytes_downloaded += len(chunk)
                
                # Check size limit during download
                if bytes_downloaded > MAX_FILE_SIZE:
                    raise ValueError(f"File exceeds size limit during download: {bytes_downloaded} bytes")
        
        temp_file.flush()
        temp_file.close()
        
        download_info["bytes_downloaded"] = bytes_downloaded
        
        # Verify file was created and has content
        if not os.path.exists(temp_file.name) or os.path.getsize(temp_file.name) == 0:
            raise OSError("Downloaded file is empty or was not created")
        
        logger.info(f"✅ Download completed: {bytes_downloaded:,} bytes → {temp_file.name}")
        
        return temp_file.name, download_info
        
    except requests.exceptions.Timeout:
        logger.error(f"Download timeout after {DOWNLOAD_TIMEOUT}s")
        raise requests.RequestException(f"Download timeout after {DOWNLOAD_TIMEOUT}s")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error downloading {url}: {e}")
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error downloading {url}: {e}", exc_info=True)
        # Clean up partial file if it exists
        if temp_file and temp_file.name and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
                logger.debug(f"Cleaned up partial download: {temp_file.name}")
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up partial file: {cleanup_error}")
        raise
    
    finally:
        # Ensure temp file is closed
        if temp_file and not temp_file.closed:
            temp_file.close()

def cleanup_temp_file(file_path: str) -> None:
    """
    Safely remove a temporary audio file.
    
    Args:
        file_path: Path to temporary file to remove
    """
    if file_path and os.path.exists(file_path):
        try:
            os.unlink(file_path)
            logger.debug(f"🧹 Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not clean up temporary file {file_path}: {e}")

# Test function for development
def test_download(url: str) -> None:
    """Test the download function with a given URL"""
    try:
        temp_path, info = download_audio_file(url)
        print(f"✅ Test download successful:")
        print(f"   File: {temp_path}")
        print(f"   Size: {info['bytes_downloaded']:,} bytes")
        print(f"   Type: {info['content_type']}")
        print(f"   Extension: {info['file_extension']}")
        
        # Clean up test file
        cleanup_temp_file(temp_path)
        
    except Exception as e:
        print(f"❌ Test download failed: {e}")

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        test_download(sys.argv[1])
    else:
        print("Usage: python audio_downloader.py <audio_url>")