"""
ImageLoader module: Load images and batches for processing.

Supports loading from parquet files and controlling sampling size.
Optimized for memory efficiency with lazy loading and selective file reading.
"""

from typing import List, Union, Any, Optional, Set, Dict
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
    - Lazy loading for memory efficiency
    - Random sampling of parquet files before reading
    - Handling both image paths and base64 encoded images
    
    Memory optimization:
    - Only reads parquet files when needed
    - Supports sampling parquet files before loading
    - Lazy loading of image paths (not actual images)
    - Images are loaded only when load() is called
    """
    
    def __init__(
        self, 
        image_dir: Union[str, Path] = None, 
        batch_size: int = 1,
        parquet_dir: Union[str, Path] = None,
        sample_size: Optional[int] = None,
        parquet_sample_size: Optional[int] = None,
        random_seed: Optional[int] = None,
        lazy_load: bool = True
    ):
        """
        Initialize ImageLoader.
        
        Args:
            image_dir: Directory containing images (for direct image loading)
            batch_size: Default batch size for batch loading
            parquet_dir: Directory containing parquet files (e.g., /mnt/tidal-alsh01/.../train/)
            sample_size: Number of images to sample from parquet files (None = use all)
            parquet_sample_size: Number of parquet files to randomly sample before reading (None = use all)
            random_seed: Random seed for reproducible sampling
            lazy_load: If True, only load parquet files when needed (memory efficient)
        """
        self.image_dir = Path(image_dir) if image_dir else None
        self.parquet_dir = Path(parquet_dir) if parquet_dir else None
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.parquet_sample_size = parquet_sample_size
        self.random_seed = random_seed
        self.lazy_load = lazy_load
        
        # Cache for loaded data
        self._image_paths: Optional[List[str]] = None
        self._image_metadata: Optional[List[dict]] = None
        self._parquet_files: Optional[List[Path]] = None
        self._loaded_parquet_files: Set[Path] = set()  # Track which files have been loaded
        
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
    
    def _discover_parquet_files(self) -> List[Path]:
        """
        Discover all parquet files in parquet_dir.
        
        Returns:
            List of parquet file paths
        """
        if not self.parquet_dir:
            return []
        
        if not self.parquet_dir.exists():
            raise FileNotFoundError(f"Parquet directory not found: {self.parquet_dir}")
        
        parquet_files = list(self.parquet_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.parquet_dir}")
        
        return parquet_files
    
    def _get_parquet_files(self) -> List[Path]:
        """Get parquet files (with optional sampling)."""
        if self._parquet_files is None:
            all_files = self._discover_parquet_files()
            
            # Sample parquet files if requested
            if self.parquet_sample_size is not None and self.parquet_sample_size < len(all_files):
                self._parquet_files = random.sample(all_files, self.parquet_sample_size)
                print(f"Sampled {self.parquet_sample_size} parquet files from {len(all_files)} total files")
            else:
                self._parquet_files = all_files
        
        return self._parquet_files
    
    def _load_single_parquet_file(self, parquet_file: Path) -> List[dict]:
        """
        Load records from a single parquet file.
        
        Supports multiple data formats:
        - OpenImages format: 'jpg' column (binary), 'conversations' column (list<struct>)
        - Standard format: 'image_path', 'image_bytes', 'image_base64', etc.
        
        Uses PyArrow directly to handle complex types (binary, list, struct) that pandas cannot convert.
        
        Args:
            parquet_file: Path to parquet file
            
        Returns:
            List of record dictionaries
        """
        try:
            # Read parquet file using PyArrow (avoid pandas conversion for complex types)
            table = pq.read_table(parquet_file, use_threads=False)
            
            if table.num_rows == 0:
                return []
            
            records = []
            file_stem = parquet_file.stem
            
            # Process row by row using PyArrow arrays (avoid pandas conversion issues)
            for row_idx in range(table.num_rows):
                record = {}
                
                # Process each column
                for col_idx, col_name in enumerate(table.column_names):
                    col = table.column(col_idx)
                    
                    # Get the value for this row
                    try:
                        value = col[row_idx]
                        
                        # Handle binary type (jpg column)
                        if col_name == 'jpg':
                            jpg_value = None
                            if isinstance(value, bytes):
                                jpg_value = value
                            elif hasattr(value, 'as_py'):
                                # PyArrow binary type
                                jpg_value = value.as_py()
                            elif value is not None:
                                # Convert to bytes
                                try:
                                    jpg_value = bytes(value)
                                except Exception:
                                    jpg_value = value

                            if isinstance(jpg_value, memoryview):
                                jpg_value = jpg_value.tobytes()
                            elif isinstance(jpg_value, bytearray):
                                jpg_value = bytes(jpg_value)
                            elif isinstance(jpg_value, list) and all(isinstance(i, int) for i in jpg_value):
                                # Some parquet writers store bytes as list<uint8>
                                jpg_value = bytes(jpg_value)
                            elif isinstance(jpg_value, dict) and "bytes" in jpg_value:
                                # Handle nested dict-like representations
                                try:
                                    jpg_value = bytes(jpg_value["bytes"])
                                except Exception:
                                    pass

                            if isinstance(jpg_value, bytes):
                                record['image_bytes'] = jpg_value
                            # Note: image_bytes will be set above, continue to next column
                            continue

                        # Handle base64-encoded image columns
                        if col_name in ('image_base64', 'image_b64', 'base64'):
                            if hasattr(value, 'as_py'):
                                record['image_bytes'] = value.as_py()
                            else:
                                record['image_bytes'] = value
                            continue
                        
                        # Handle list/struct types (conversations column) - convert to Python native
                        # Check if column type is list or struct using PyArrow type checking
                        col_type = col.type
                        is_list_type = str(col_type).startswith('list')
                        is_struct_type = str(col_type).startswith('struct')
                        
                        if is_list_type or is_struct_type:
                            if hasattr(value, 'as_py'):
                                record[col_name] = value.as_py()
                            elif value is not None:
                                # Try to convert nested structures
                                record[col_name] = self._convert_arrow_value(value)
                            else:
                                record[col_name] = None
                        
                        # Handle other types - try to convert to Python native
                        elif value is not None:
                            if hasattr(value, 'as_py'):
                                record[col_name] = value.as_py()
                            else:
                                record[col_name] = value
                        else:
                            record[col_name] = None
                            
                    except Exception as e:
                        # Skip problematic values
                        record[col_name] = None
                
                # Must have image data (jpg column provides image_bytes)
                if 'image_bytes' not in record:
                    # Try alternative image sources
                    if 'image_path' in record and record['image_path']:
                        pass  # Has image_path, OK
                    elif 'path' in record and record['path']:
                        record['image_path'] = record['path']
                    elif 'file_path' in record and record['file_path']:
                        record['image_path'] = record['file_path']
                    elif 'image_url' in record and record['image_url']:
                        record['image_path'] = record['image_url']
                    else:
                        # No image data found, skip this record
                        continue
                
                # Generate image ID if not present
                if 'image_id' not in record or not record['image_id']:
                    if 'image_path' in record and record['image_path']:
                        record['image_id'] = Path(record['image_path']).stem
                    else:
                        record['image_id'] = f"{file_stem}_{row_idx}"
                
                records.append(record)
            
            return records
            
        except Exception as e:
            print(f"Warning: Error reading {parquet_file}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _convert_arrow_value(self, value):
        """
        Convert PyArrow value to Python native type (recursive for nested structures).
        
        Args:
            value: PyArrow value (can be nested)
            
        Returns:
            Python native value
        """
        import pyarrow as pa
        
        if value is None:
            return None
        
        # Use as_py() if available (handles most conversions)
        if hasattr(value, 'as_py'):
            return value.as_py()
        
        # Handle list types
        if isinstance(value, (list, pa.lib.ListArray)):
            return [self._convert_arrow_value(item) for item in value]
        
        # Handle dict/struct types
        if isinstance(value, dict):
            return {k: self._convert_arrow_value(v) for k, v in value.items()}
        
        # Fallback: return as-is
        return value
    
    def _load_parquet_files(self, force_reload: bool = False) -> List[dict]:
        """
        Load image paths and metadata from parquet files.
        
        Uses lazy loading: only loads files that haven't been loaded yet.
        
        Args:
            force_reload: If True, reload all files even if already loaded
            
        Returns:
            List of dictionaries containing image metadata (path, id, etc.)
        """
        if not HAS_PARQUET:
            raise ImportError(
                "pandas and pyarrow are required for parquet file support. "
                "Install with: pip install pandas pyarrow"
            )
        
        # Get parquet files to load
        parquet_files = self._get_parquet_files()
        
        if not parquet_files:
            return []
        
        # Determine which files to load
        if force_reload:
            files_to_load = parquet_files
            all_records = []
        else:
            # Only load files that haven't been loaded yet
            files_to_load = [f for f in parquet_files if f not in self._loaded_parquet_files]
            # Start with already loaded records
            all_records = list(self._image_metadata) if self._image_metadata else []
        
        # Load new files
        for parquet_file in files_to_load:
            records = self._load_single_parquet_file(parquet_file)
            all_records.extend(records)
            self._loaded_parquet_files.add(parquet_file)
        
        return all_records

    def load_parquet_files(self, parquet_files: List[Path]) -> List[dict]:
        """
        Load specific parquet files and append to metadata cache.
        
        Args:
            parquet_files: List of parquet file paths to load
        
        Returns:
            List of record dictionaries loaded from these files
        """
        if not HAS_PARQUET:
            raise ImportError(
                "pandas and pyarrow are required for parquet file support. "
                "Install with: pip install pandas pyarrow"
            )
        
        if not parquet_files:
            return []
        
        all_records = []
        for parquet_file in parquet_files:
            if parquet_file in self._loaded_parquet_files:
                continue
            records = self._load_single_parquet_file(parquet_file)
            all_records.extend(records)
            self._loaded_parquet_files.add(parquet_file)
        
        if all_records:
            if self._image_metadata is None:
                self._image_metadata = []
            self._image_metadata.extend(all_records)
        
        return all_records

    def get_parquet_files(self) -> List[Path]:
        """Public accessor for parquet file list (with optional sampling)."""
        return self._get_parquet_files()

    def _records_to_image_paths(self, records: List[dict]) -> List[str]:
        """Convert records to image path identifiers."""
        image_paths = []
        for record in records:
            if 'image_path' in record:
                image_paths.append(record['image_path'])
            elif 'image_bytes' in record:
                image_id = record.get('image_id', f"image_{len(image_paths)}")
                image_paths.append(f"<bytes:{image_id}>")
        return image_paths
    
    def get_image_paths(self, force_reload: bool = False) -> List[str]:
        """
        Get list of image paths from parquet files.
        
        Uses lazy loading for memory efficiency.
        
        Args:
            force_reload: If True, reload from parquet files even if cached
            
        Returns:
            List of image paths (or placeholder strings for base64 images)
        """
        # If already loaded and not forcing reload, return cached
        if self._image_paths is not None and not force_reload:
            return self._image_paths
        
        # Load parquet files (lazy, only if needed)
        records = self._load_parquet_files(force_reload=force_reload)
        
        if not records:
            self._image_paths = []
            self._image_metadata = []
            return []
        
        # Extract image paths/identifiers
        image_paths = self._records_to_image_paths(records)
        
        # Apply image-level sampling if requested
        if self.sample_size is not None and self.sample_size < len(image_paths):
            # Sample records and corresponding paths together
            sampled_indices = random.sample(range(len(image_paths)), self.sample_size)
            image_paths = [image_paths[i] for i in sampled_indices]
            records = [records[i] for i in sampled_indices]
        
        self._image_paths = image_paths
        self._image_metadata = records
        
        if len(image_paths) > 0:
            print(f"Loaded {len(image_paths)} image records from {len(self._loaded_parquet_files)} parquet file(s)")
        
        return self._image_paths
    
    def load(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load a single image.
        
        Supports:
        - Direct file path loading
        - Loading from image bytes/base64 (if path starts with "<bytes:")
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
                    if record.get('image_id') == image_id:
                        image_bytes = None
                        if 'image_bytes' in record:
                            image_bytes = record['image_bytes']
                        elif 'image_base64' in record:
                            image_bytes = record['image_base64']
                        elif 'image_b64' in record:
                            image_bytes = record['image_b64']
                        elif 'base64' in record:
                            image_bytes = record['base64']

                        if image_bytes is not None:
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
                                except Exception as e:
                                    raise ValueError(f"Could not decode image bytes for {image_id}: {e}")
            
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
    
    def get_image_metadata(self, image_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Get metadata for an image (excluding image data fields).
        
        Args:
            image_path: Path to the image file or "<bytes:image_id>" format
            
        Returns:
            Dictionary containing metadata (e.g., conversations) or None if not found
        """
        image_path_str = str(image_path)
        image_id = self.get_image_id(image_path)
        
        # Find the record with this image_id
        if self._image_metadata:
            for record in self._image_metadata:
                if record.get('image_id') == image_id:
                    # Return metadata excluding image data fields
                    metadata = {}
                    exclude_fields = {'image_path', 'path', 'file_path', 'image_url', 
                                    'image_bytes', 'image_base64', 'jpg', 'image_id', 'id'}
                    for key, value in record.items():
                        if key not in exclude_fields:
                            metadata[key] = value
                    return metadata if metadata else None
        
        return None
    
    def get_all_image_paths(self) -> List[str]:
        """
        Get all image paths (with sampling applied if sample_size is set).
        
        This is a convenience method that calls get_image_paths().
        Uses lazy loading - only loads parquet files when needed.
        
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
    
    def set_parquet_sample_size(self, parquet_sample_size: Optional[int]):
        """
        Update parquet file sample size.
        
        Args:
            parquet_sample_size: Number of parquet files to sample (None = use all)
        """
        self.parquet_sample_size = parquet_sample_size
        self._parquet_files = None  # Clear cache
        self._image_paths = None
        self._image_metadata = None
        self._loaded_parquet_files.clear()
    
    def sample_images_without_replacement(
        self, 
        batch_size: int, 
        exclude_paths: Optional[Set[str]] = None
    ) -> List[str]:
        """
        Sample a batch of images without replacement from all available parquet files.
        
        This method loads all parquet files (if not already loaded) and samples
        a batch of images excluding those already processed.
        
        Args:
            batch_size: Number of images to sample
            exclude_paths: Set of image paths to exclude (already processed)
            
        Returns:
            List of sampled image paths
        """
        if exclude_paths is None:
            exclude_paths = set()
        
        # Load all parquet files to get all available images
        all_records = self._load_parquet_files(force_reload=False)
        # Ensure metadata cache is available for load() when using <bytes:...>
        self._image_metadata = all_records
        
        if not all_records:
            return []
        
        # Extract all image paths
        all_image_paths = []
        for record in all_records:
            if 'image_path' in record:
                path = record['image_path']
                if path not in exclude_paths:
                    all_image_paths.append(path)
            elif 'image_bytes' in record:
                image_id = record.get('image_id', f"image_{len(all_image_paths)}")
                path = f"<bytes:{image_id}>"
                if path not in exclude_paths:
                    all_image_paths.append(path)
        
        # Sample without replacement
        if batch_size >= len(all_image_paths):
            return all_image_paths
        
        sampled_paths = random.sample(all_image_paths, batch_size)
        return sampled_paths
