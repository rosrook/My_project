"""
ImageLoader module: Load images and batches for processing.

Supports loading from parquet files and controlling sampling size.
"""

from typing import List, Union, Any, Optional
from pathlib import Path
from PIL import Image
import random
import io

try:
    import pandas as pd
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    pd = None
    pq = None


class ImageLoader:
    """
    Load images and batches for probing factor generation.
    
    Supports:
    - Loading from parquet files (OpenImages dataset format)
    - Controlling sampling size
    - Single image loading and batch processing
    """
    
    def __init__(
        self, 
        image_dir: Union[str, Path] = None, 
        batch_size: int = 1,
        parquet_dir: Union[str, Path] = None,
        sample_size: Optional[int] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize ImageLoader.
        
        Args:
            image_dir: Directory containing images (for direct image loading)
            batch_size: Default batch size for batch loading
            parquet_dir: Directory containing parquet files (e.g., /mnt/tidal-alsh01/dataset/.../train/)
            sample_size: Number of images to sample from parquet files (None = use all)
            random_seed: Random seed for reproducible sampling
        """
        self.image_dir = Path(image_dir) if image_dir else None
        self.parquet_dir = Path(parquet_dir) if parquet_dir else None
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.random_seed = random_seed
        
        # Cache for loaded image paths from parquet files
        self._image_paths: Optional[List[str]] = None
        self._image_metadata: Optional[List[dict]] = None
        
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
    
    def _load_parquet_files(self) -> List[dict]:
        """
        Load image paths and metadata from all parquet files in parquet_dir.
        
        Returns:
            List of dictionaries containing image metadata (path, id, etc.)
        """
        if not self.parquet_dir:
            return []
        
        if not self.parquet_dir.exists():
            raise FileNotFoundError(f"Parquet directory not found: {self.parquet_dir}")
        
        if not HAS_PARQUET:
            raise ImportError(
                "pandas and pyarrow are required for parquet file support. "
                "Install with: pip install pandas pyarrow"
            )
        
        # Find all parquet files
        parquet_files = list(self.parquet_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.parquet_dir}")
        
        print(f"Found {len(parquet_files)} parquet files in {self.parquet_dir}")
        
        all_records = []
        for parquet_file in parquet_files:
            try:
                # Read parquet file
                table = pq.read_table(parquet_file)
                df = table.to_pandas()
                
                # Convert to list of records
                # Handle different possible column names
                for _, row in df.iterrows():
                    record = {}
                    
                    # Try to extract image path (common column names)
                    if 'image_path' in df.columns:
                        record['image_path'] = row['image_path']
                    elif 'path' in df.columns:
                        record['image_path'] = row['path']
                    elif 'file_path' in df.columns:
                        record['image_path'] = row['file_path']
                    else:
                        # If no path column, skip (or try to extract from image_bytes)
                        if 'image_bytes' not in df.columns:
                            continue
                        record['image_bytes'] = row['image_bytes']
                    
                    # Try to extract image ID
                    if 'image_id' in df.columns:
                        record['image_id'] = row['image_id']
                    elif 'id' in df.columns:
                        record['image_id'] = row['id']
                    elif 'image_path' in record:
                        # Use path stem as ID
                        record['image_id'] = Path(record['image_path']).stem
                    
                    # Store image bytes if available
                    if 'image_bytes' in df.columns:
                        record['image_bytes'] = row['image_bytes']
                    
                    # Store all other columns as metadata
                    for col in df.columns:
                        if col not in ['image_path', 'path', 'file_path', 'image_id', 'id', 'image_bytes']:
                            record[col] = row[col]
                    
                    all_records.append(record)
                    
            except Exception as e:
                print(f"Warning: Error reading {parquet_file}: {e}")
                continue
        
        print(f"Loaded {len(all_records)} image records from parquet files")
        return all_records
    
    def get_image_paths(self, force_reload: bool = False) -> List[str]:
        """
        Get list of image paths from parquet files.
        
        Args:
            force_reload: If True, reload from parquet files even if cached
            
        Returns:
            List of image paths
        """
        if self._image_paths is not None and not force_reload:
            return self._image_paths
        
        if not self.parquet_dir:
            return []
        
        records = self._load_parquet_files()
        
        # Extract image paths
        image_paths = []
        for record in records:
            if 'image_path' in record:
                image_paths.append(record['image_path'])
            elif 'image_bytes' in record:
                # If only bytes are available, we'll handle it in load()
                # For now, use a placeholder path
                image_id = record.get('image_id', f"image_{len(image_paths)}")
                image_paths.append(f"<bytes:{image_id}>")
        
        # Apply sampling if requested
        if self.sample_size is not None and self.sample_size < len(image_paths):
            image_paths = random.sample(image_paths, self.sample_size)
            print(f"Sampled {self.sample_size} images from {len(records)} total records")
        
        self._image_paths = image_paths
        self._image_metadata = records
        
        return self._image_paths
    
    def load(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load a single image.
        
        Supports:
        - Direct file path loading
        - Loading from image bytes (if path starts with "<bytes:")
        - Relative paths relative to image_dir
        
        Args:
            image_path: Path to the image file or "<bytes:image_id>" format
            
        Returns:
            Loaded PIL Image object (RGB format)
        """
        image_path_str = str(image_path)
        
        # Handle image bytes (from parquet)
        if image_path_str.startswith("<bytes:"):
            image_id = image_path_str[7:-1]  # Remove "<bytes:" and ">"
            
            # Find the record with this image_id
            if self._image_metadata:
                for record in self._image_metadata:
                    if record.get('image_id') == image_id and 'image_bytes' in record:
                        image_bytes = record['image_bytes']
                        # Handle both bytes and base64 encoded strings
                        if isinstance(image_bytes, bytes):
                            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                            return image
                        elif isinstance(image_bytes, str):
                            # Try base64 decoding
                            try:
                                import base64
                                decoded_bytes = base64.b64decode(image_bytes)
                                image = Image.open(io.BytesIO(decoded_bytes)).convert('RGB')
                                return image
                            except Exception:
                                raise ValueError(f"Could not decode image bytes for {image_id}")
            
            raise ValueError(f"Image bytes not found for ID: {image_id}")
        
        # Handle regular file path
        image_path = Path(image_path_str)
        
        # If relative path and image_dir is set, join them
        if not image_path.is_absolute() and self.image_dir:
            image_path = self.image_dir / image_path
        elif not image_path.is_absolute():
            # Try to find in parquet metadata
            if self._image_metadata:
                for record in self._image_metadata:
                    if record.get('image_path') == image_path_str:
                        # Recursively call with full path
                        return self.load(record['image_path'])
        
        # Load image
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            raise IOError(f"Error loading image {image_path}: {e}")
    
    def load_batch(self, image_paths: List[Union[str, Path]]) -> List[Image.Image]:
        """
        Load a batch of images.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of loaded PIL Image objects
        """
        images = []
        for img_path in image_paths:
            try:
                image = self.load(img_path)
                images.append(image)
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")
                # Optionally skip failed images or raise
                # For now, we'll skip and continue
                continue
        return images
    
    def get_image_id(self, image_path: Union[str, Path]) -> str:
        """
        Extract image ID from image path.
        
        Args:
            image_path: Path to the image file or "<bytes:image_id>" format
            
        Returns:
            Image ID string (filename without extension or image_id from metadata)
        """
        image_path_str = str(image_path)
        
        # Handle image bytes format
        if image_path_str.startswith("<bytes:"):
            return image_path_str[7:-1]  # Return the image_id part
        
        # Try to find in metadata first
        if self._image_metadata:
            for record in self._image_metadata:
                if record.get('image_path') == image_path_str and 'image_id' in record:
                    return str(record['image_id'])
        
        # Fall back to extracting from path
        path = Path(image_path_str)
        return path.stem
    
    def get_all_image_paths(self) -> List[str]:
        """
        Get all image paths (with sampling applied if sample_size is set).
        
        This is a convenience method that calls get_image_paths().
        
        Returns:
            List of image paths
        """
        return self.get_image_paths()
    
    def set_sample_size(self, sample_size: Optional[int]):
        """
        Update sample size and reload paths.
        
        Args:
            sample_size: New sample size (None = use all)
        """
        self.sample_size = sample_size
        self._image_paths = None  # Clear cache to force reload
        self._image_metadata = None
