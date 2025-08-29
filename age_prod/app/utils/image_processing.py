import os
import tempfile
import shutil
import uuid
from typing import Optional
import aiofiles
from fastapi import UploadFile
import logging

logger = logging.getLogger(__name__)

async def save_uploaded_file(file: UploadFile, temp_dir: str = "/tmp") -> str:
    """
    Save uploaded file to temporary location
    
    Args:
        file: FastAPI UploadFile object
        temp_dir: Temporary directory path
        
    Returns:
        Path to saved temporary file
    """
    try:
        # Generate unique filename
        file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
        temp_filename = f"temp_{uuid.uuid4().hex}.{file_extension}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # Ensure temp directory exists
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save file asynchronously
        async with aiofiles.open(temp_path, 'wb') as temp_file:
            content = await file.read()
            await temp_file.write(content)
        
        logger.info(f"Uploaded file saved to: {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise Exception(f"Failed to save uploaded file: {str(e)}")

def cleanup_temp_files(*file_paths: str) -> None:
    """
    Clean up temporary files
    
    Args:
        *file_paths: Variable number of file paths to delete
    """
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup file {file_path}: {e}")

def validate_image_format(filename: str) -> bool:
    """
    Validate if the file has a supported image format
    
    Args:
        filename: Name of the file to validate
        
    Returns:
        True if format is supported, False otherwise
    """
    if not filename:
        return False
        
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    file_extension = os.path.splitext(filename.lower())[1]
    
    return file_extension in supported_formats

def ensure_results_directory(results_dir: str = "/app/results") -> None:
    """
    Ensure results directory exists
    
    Args:
        results_dir: Path to results directory
    """
    try:
        os.makedirs(results_dir, exist_ok=True)
        logger.info(f"Results directory ensured: {results_dir}")
    except Exception as e:
        logger.error(f"Failed to create results directory: {e}")
        raise Exception(f"Cannot create results directory: {str(e)}")

async def save_multiple_uploaded_files(files: list, temp_dir: str = "/tmp") -> list:
    """
    Save multiple uploaded files to temporary locations
    
    Args:
        files: List of FastAPI UploadFile objects
        temp_dir: Temporary directory path
        
    Returns:
        List of paths to saved temporary files
    """
    saved_paths = []
    
    for file in files:
        try:
            temp_path = await save_uploaded_file(file, temp_dir)
            saved_paths.append(temp_path)
        except Exception as e:
            logger.error(f"Failed to save file {file.filename}: {e}")
            # Cleanup already saved files on error
            cleanup_temp_files(*saved_paths)
            raise Exception(f"Failed to save multiple files: {str(e)}")
    
    return saved_paths

def get_file_size(file_path: str) -> Optional[int]:
    """
    Get file size in bytes
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes, or None if file doesn't exist
    """
    try:
        if os.path.exists(file_path):
            return os.path.getsize(file_path)
        return None
    except Exception:
        return None

def create_temp_directory() -> str:
    """
    Create a temporary directory for processing
    
    Returns:
        Path to created temporary directory
    """
    try:
        temp_dir = tempfile.mkdtemp(prefix="age_prod_")
        logger.info(f"Created temporary directory: {temp_dir}")
        return temp_dir
    except Exception as e:
        logger.error(f"Failed to create temporary directory: {e}")
        raise Exception(f"Cannot create temporary directory: {str(e)}")

def cleanup_temp_directory(temp_dir: str) -> None:
    """
    Clean up entire temporary directory
    
    Args:
        temp_dir: Path to temporary directory to remove
    """
    try:
        if os.path.exists(temp_dir) and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temporary directory {temp_dir}: {e}")