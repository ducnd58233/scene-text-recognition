import os
import gdown
import zipfile
import hashlib
import logging
import mimetypes
import requests
import time
from pathlib import Path
from typing import Union, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm
from urllib.parse import urlparse

class GDriveDownloader:
    """
    Optimized dataset downloader with separate download and extract functionalities.
    """
    
    MIME_TYPES = {
        'zip': ['application/zip', 'application/x-zip-compressed'],
        'tar': ['application/x-tar'],
        'gzip': ['application/gzip', 'application/x-gzip'],
        '7z': ['application/x-7z-compressed'],
        'rar': ['application/x-rar-compressed']
    }
    
    def __init__(
        self,
        cache_dir: Union[str, Path] = None,
        log_level: int = logging.INFO
    ):
        """
        Initialize the downloader.
        
        Args:
            cache_dir: Directory for caching downloaded files
            log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.gdrive_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging(log_level)

    def _setup_logging(self, log_level: int) -> None:
        """Configure logging with custom format."""
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)

    def _get_mime_type(self, url: str) -> Optional[str]:
        """
        Get MIME type of the file from URL or content headers.
        
        Args:
            url: File URL
            
        Returns:
            Optional[str]: MIME type if detected, None otherwise
        """
        try:
            # First try to get from URL
            mime_type, _ = mimetypes.guess_type(url)
            if mime_type:
                return mime_type

            # If not found, try to get from headers
            headers = requests.head(url, allow_redirects=True).headers
            return headers.get('content-type')
        except Exception as e:
            self.logger.warning(f"Failed to detect MIME type: {e}")
            return None

    def _is_compressed_file(self, mime_type: Optional[str]) -> bool:
        """
        Check if the file is a compressed archive.
        
        Args:
            mime_type: MIME type of the file
            
        Returns:
            bool: True if file is compressed archive
        """
        if not mime_type:
            return False
        
        return any(
            mime_type in mime_types 
            for mime_types in self.MIME_TYPES.values()
        )

    def download(
        self,
        gdrive_url: str,
        output_path: Union[str, Path],
        force_download: bool = False,
        validate_compression: bool = True,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Download file from Google Drive with validation.
        
        Args:
            gdrive_url: Google Drive URL
            output_path: Path to save the file
            force_download: Force download even if file exists
            validate_compression: Check if file is compressed archive
            
        Returns:
            Tuple[bool, str, Optional[str]]: (success, message, mime_type)
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if file already exists
            if not force_download and output_path.exists():
                mime_type = mimetypes.guess_type(str(output_path))[0]
                return True, "File already exists", mime_type

            # Extract file ID and create download URL
            if 'drive.google.com' in gdrive_url:
                if 'file/d/' in gdrive_url:
                    file_id = gdrive_url.split('file/d/')[1].split('/')[0]
                elif 'id=' in gdrive_url:
                    file_id = gdrive_url.split('id=')[1].split('&')[0]
                else:
                    return False, "Invalid Google Drive URL format", None
                
                download_url = f"https://drive.google.com/uc?id={file_id}"
            else:
                download_url = gdrive_url

            # Download file with progress bar
            self.logger.info(f"Downloading file to {output_path}")
            success = gdown.download(download_url, str(output_path), quiet=False)
            
            if not success:
                return False, "Download failed", None

            # Validate file exists and is not empty
            if not output_path.exists() or output_path.stat().st_size == 0:
                return False, "Downloaded file is empty or missing", None

            # Get and validate MIME type
            mime_type = self._get_mime_type(str(output_path))
            if validate_compression and not self._is_compressed_file(mime_type):
                return False, "Downloaded file is not a compressed archive", mime_type

            return True, "Download successful", mime_type

        except Exception as e:
            self.logger.error(f"Download error: {str(e)}")
            return False, f"Download failed: {str(e)}", None

    def extract(
        self,
        archive_path: Union[str, Path],
        extract_path: Union[str, Path],
        remove_archive: bool = False,
        num_threads: int = 4
    ) -> Tuple[bool, str]:
        """
        Extract compressed archive with progress tracking.
        
        Args:
            archive_path: Path to compressed archive
            extract_path: Extraction destination
            remove_archive: Remove archive after extraction
            num_threads: Number of threads for extraction
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            archive_path = Path(archive_path)
            extract_path = Path(extract_path)
            extract_path.mkdir(parents=True, exist_ok=True)

            if not archive_path.exists():
                return False, "Archive file not found"

            mime_type = mimetypes.guess_type(str(archive_path))[0]
            if not self._is_compressed_file(mime_type):
                return False, "File is not a compressed archive"

            # Extract with progress bar
            self.logger.info(f"Extracting to {extract_path}")
            with zipfile.ZipFile(archive_path) as zf:
                total = sum(file.file_size for file in zf.filelist)
                extracted = 0
                
                with tqdm(total=total, unit='B', unit_scale=True, desc="Extracting") as pbar:
                    for file in zf.filelist:
                        zf.extract(file, extract_path)
                        extracted += file.file_size
                        pbar.update(file.file_size)

            # Validate extraction
            if not any(extract_path.iterdir()):
                return False, "No files found after extraction"

            # Remove archive if requested
            if remove_archive:
                archive_path.unlink()
                self.logger.info(f"Removed archive file: {archive_path}")

            return True, "Extraction successful"

        except zipfile.BadZipFile:
            return False, "Invalid or corrupted zip file"
        except Exception as e:
            self.logger.error(f"Extraction error: {str(e)}")
            return False, f"Extraction failed: {str(e)}"

    def download_and_extract(
        self,
        gdrive_url: str,
        extract_dir: Union[str, Path],
        keep_zip: bool = False,
        force_download: bool = False,
        num_threads: int = 4
    ) -> Tuple[bool, str]:
        """
        Combined download and extract functionality.
        
        Args:
            gdrive_url: Google Drive URL
            extract_dir: Extraction destination
            keep_zip: Keep zip file after extraction
            force_download: Force download even if file exists
            num_threads: Number of threads for extraction
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        # Create temporary zip path
        temp_zip = self.cache_dir / f"temp_{int(time.time())}.zip"
        
        # Download
        success, message, mime_type = self.download(
            gdrive_url=gdrive_url,
            output_path=temp_zip,
            force_download=force_download,
            validate_compression=True
        )
        
        if not success:
            return False, f"Download phase failed: {message}"

        # Extract
        success, message = self.extract(
            archive_path=temp_zip,
            extract_path=extract_dir,
            remove_archive=not keep_zip,
            num_threads=num_threads
        )
        
        if not success:
            # Clean up zip if extraction failed
            if temp_zip.exists():
                temp_zip.unlink()
            return False, f"Extraction phase failed: {message}"

        return True, "Download and extraction completed successfully"

    def cleanup_cache(self, max_age_days: int = 30) -> None:
        """Clean up old cache files."""
        try:
            current_time = time.time()
            for cache_file in self.cache_dir.glob("*.*"):
                if (current_time - cache_file.stat().st_mtime) > (max_age_days * 86400):
                    cache_file.unlink()
                    self.logger.info(f"Removed old cache file: {cache_file}")
        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Initialize downloader
    downloader = GDriveDownloader(cache_dir="./cache")
    
    # Example 1: Separate download and extract
    success, message, mime_type = downloader.download(
        gdrive_url="YOUR_GDRIVE_URL",
        output_path="dataset.zip",
        validate_compression=True
    )
    
    if success:
        success, message = downloader.extract(
            archive_path="dataset.zip",
            extract_path="./extracted_data",
            remove_archive=True
        )
    
    # Example 2: Combined download and extract
    success, message = downloader.download_and_extract(
        gdrive_url="YOUR_GDRIVE_URL",
        extract_dir="./extracted_data",
        keep_zip=False
    )