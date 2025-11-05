"""
Data preparation script for CARLA self-driving dataset.
Converts parquet files to organized numpy/jpg files with metadata index.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import json

# Try to import pyarrow for parquet support
try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

# Try to import datasets library
try:
    from datasets import load_dataset, load_from_disk
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


# Key mapping for dataset variations
DEFAULT_KEY_MAPPING = {
    'image_front': ['image_front', 'rgb_front', 'front_image', 'image'],
    'lidar': ['lidar', 'lidar_points', 'point_cloud'],
    'steer': ['steer', 'steering', 'steering_angle'],
    'throttle': ['throttle', 'acceleration', 'accel'],
    'brake': ['brake', 'braking'],
    'speed_kmh': ['speed_kmh', 'speed', 'velocity'],
    'map_name': ['map_name', 'map', 'town'],
    'timestamp': ['timestamp', 'time', 'frame_id']
}


def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def find_key(data: Dict, possible_keys: List[str]) -> Optional[str]:
    """Find the first matching key from a list of possible keys."""
    for key in possible_keys:
        if key in data:
            return key
    return None


def create_bev_projection(
    lidar_points: np.ndarray,
    x_range: tuple = (-50, 50),
    y_range: tuple = (-50, 50),
    grid_size: int = 256,
    z_min: float = -3.0,
    z_max: float = 5.0
) -> np.ndarray:
    """
    Create Bird's Eye View projection from LiDAR points.
    
    Args:
        lidar_points: Nx4 array of [x, y, z, intensity]
        x_range: (min, max) range for x-axis in meters
        y_range: (min, max) range for y-axis in meters
        grid_size: Size of output grid (grid_size x grid_size)
        z_min: Minimum z value to consider
        z_max: Maximum z value to consider
    
    Returns:
        BEV projection as (grid_size, grid_size) array
    """
    if len(lidar_points) == 0:
        return np.zeros((grid_size, grid_size), dtype=np.float32)
    
    # Filter by height
    mask = (lidar_points[:, 2] >= z_min) & (lidar_points[:, 2] <= z_max)
    points = lidar_points[mask]
    
    if len(points) == 0:
        return np.zeros((grid_size, grid_size), dtype=np.float32)
    
    # Create BEV grid
    x_bins = np.linspace(x_range[0], x_range[1], grid_size + 1)
    y_bins = np.linspace(y_range[0], y_range[1], grid_size + 1)
    
    # Bin points
    x_indices = np.digitize(points[:, 0], x_bins) - 1
    y_indices = np.digitize(points[:, 1], y_bins) - 1
    
    # Filter out-of-range points
    valid = (x_indices >= 0) & (x_indices < grid_size) & \
            (y_indices >= 0) & (y_indices < grid_size)
    
    x_indices = x_indices[valid]
    y_indices = y_indices[valid]
    heights = points[valid, 2]
    
    # Create height map (max height in each cell)
    bev = np.zeros((grid_size, grid_size), dtype=np.float32)
    for x_idx, y_idx, h in zip(x_indices, y_indices, heights):
        bev[y_idx, x_idx] = max(bev[y_idx, x_idx], h)
    
    return bev


def process_sample(
    sample: Dict[str, Any],
    idx: int,
    split: str,
    output_dir: Path,
    key_mapping: Dict[str, str],
    generate_bev: bool = False,
    image_quality: int = 85
) -> Optional[Dict[str, Any]]:
    """
    Process a single sample and save to disk.
    
    Returns:
        Metadata dictionary for this sample, or None if processing failed
    """
    try:
        # Extract image
        image_key = key_mapping.get('image_front')
        if image_key not in sample:
            logging.warning(f"Sample {idx}: Missing image key '{image_key}'")
            return None
        
        image_data = sample[image_key]
        
        # Handle different image formats
        if isinstance(image_data, dict) and 'bytes' in image_data:
            # Image stored as dict with bytes
            from io import BytesIO
            image = Image.open(BytesIO(image_data['bytes']))
        elif isinstance(image_data, bytes):
            # Raw bytes
            from io import BytesIO
            image = Image.open(BytesIO(image_data))
        elif isinstance(image_data, Image.Image):
            # Already a PIL image
            image = image_data
        else:
            # Try to convert to PIL Image from array
            try:
                image = Image.fromarray(np.array(image_data))
            except Exception as e:
                logging.error(f"Sample {idx}: Cannot convert image data to PIL Image: {e}")
                return None
        
        # Convert RGBA to RGB if needed (JPEG doesn't support RGBA)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Save image
        image_path = output_dir / 'images' / split / f'{idx:06d}.jpg'
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path, 'JPEG', quality=image_quality)
        
        # Extract LiDAR
        lidar_key = key_mapping.get('lidar')
        if lidar_key not in sample:
            logging.warning(f"Sample {idx}: Missing lidar key '{lidar_key}'")
            return None
        
        lidar_data = sample[lidar_key]
        
        # Handle different LiDAR formats
        if isinstance(lidar_data, np.ndarray):
            # If it's an array of objects (arrays), convert to 2D array
            if lidar_data.dtype == object:
                try:
                    lidar = np.vstack([np.array(point) for point in lidar_data])
                except Exception as e:
                    logging.error(f"Sample {idx}: Cannot stack LiDAR points: {e}")
                    return None
            else:
                lidar = lidar_data
        elif isinstance(lidar_data, list):
            lidar = np.array(lidar_data)
            lidar = np.array(lidar, dtype=np.float32)
        elif not isinstance(lidar, np.ndarray):
            lidar = np.array(lidar, dtype=np.float32)
        
        # Ensure lidar is Nx4
        if lidar.ndim == 1:
            # Might be flattened
            if len(lidar) % 4 == 0:
                lidar = lidar.reshape(-1, 4)
            else:
                logging.warning(f"Sample {idx}: Invalid lidar shape")
                return None
        
        if lidar.shape[1] != 4:
            logging.warning(f"Sample {idx}: Lidar should have 4 columns, got {lidar.shape[1]}")
            return None
        
        # Save LiDAR
        lidar_path = output_dir / 'lidar' / split / f'{idx:06d}.npy'
        lidar_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(lidar_path, lidar.astype(np.float32))
        
        # Generate BEV if requested
        bev_path = None
        if generate_bev:
            bev = create_bev_projection(lidar)
            bev_path = output_dir / 'bev' / split / f'{idx:06d}.npy'
            bev_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(bev_path, bev)
        
        # Extract control signals and metadata
        metadata = {
            'idx': idx,
            'steer': float(sample.get(key_mapping.get('steer'), 0.0)),
            'throttle': float(sample.get(key_mapping.get('throttle'), 0.0)),
            'brake': float(sample.get(key_mapping.get('brake'), 0.0)),
            'speed_kmh': float(sample.get(key_mapping.get('speed_kmh'), 0.0)),
            'path_to_image': str(image_path.relative_to(output_dir)),
            'path_to_lidar': str(lidar_path.relative_to(output_dir)),
        }
        
        # Optional metadata
        if generate_bev and bev_path:
            metadata['path_to_bev'] = str(bev_path.relative_to(output_dir))
        
        map_key = key_mapping.get('map_name')
        if map_key and map_key in sample:
            metadata['map_name'] = str(sample[map_key])
        
        timestamp_key = key_mapping.get('timestamp')
        if timestamp_key and timestamp_key in sample:
            metadata['timestamp'] = float(sample[timestamp_key])
        
        return metadata
        
    except Exception as e:
        logging.error(f"Error processing sample {idx}: {e}")
        return None


def prepare_from_parquet(
    data_dir: Path,
    output_dir: Path,
    split: str,
    max_samples: Optional[int] = None,
    generate_bev: bool = False,
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    Prepare data from parquet files.
    
    Args:
        data_dir: Directory containing parquet files
        output_dir: Output directory for processed data
        split: Data split name ('train', 'validation', 'test')
        max_samples: Maximum number of samples to process
        generate_bev: Whether to generate BEV projections
        debug: Debug mode
    
    Returns:
        List of metadata dictionaries
    """
    logger = logging.getLogger(__name__)
    
    # Find parquet files
    split_dir = data_dir / f'partial-{split}'
    if not split_dir.exists():
        # Try alternative naming
        split_dir = data_dir / split
    
    if not split_dir.exists():
        logger.error(f"Split directory not found: {split_dir}")
        return []
    
    parquet_files = sorted(split_dir.glob('*.parquet'))
    if not parquet_files:
        logger.error(f"No parquet files found in {split_dir}")
        return []
    
    logger.info(f"Found {len(parquet_files)} parquet files for split '{split}'")
    
    # Read parquet files and infer key mapping
    metadata_list = []
    sample_idx = 0
    
    for parquet_file in parquet_files:
        logger.info(f"Processing {parquet_file.name}...")
        
        # Read parquet file
        df = pd.read_parquet(parquet_file)
        
        # Infer key mapping from first file
        if sample_idx == 0:
            key_mapping = {}
            for standard_key, possible_keys in DEFAULT_KEY_MAPPING.items():
                found_key = find_key(df.columns.tolist(), possible_keys)
                if found_key:
                    key_mapping[standard_key] = found_key
                    logger.debug(f"Mapped '{standard_key}' -> '{found_key}'")
            
            # Verify essential keys
            essential_keys = ['image_front', 'lidar', 'steer', 'throttle', 'brake']
            missing = [k for k in essential_keys if k not in key_mapping]
            if missing:
                logger.error(f"Missing essential keys: {missing}")
                logger.error(f"Available columns: {df.columns.tolist()}")
                return []
        
        # Process each row
        for row_idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{split}/{parquet_file.name}"):
            if max_samples and sample_idx >= max_samples:
                break
            
            sample = row.to_dict()
            metadata = process_sample(
                sample, sample_idx, split, output_dir,
                key_mapping, generate_bev
            )
            
            if metadata:
                metadata_list.append(metadata)
            
            sample_idx += 1
        
        if max_samples and sample_idx >= max_samples:
            break
    
    logger.info(f"Processed {len(metadata_list)} samples for split '{split}'")
    return metadata_list


def prepare_from_hf_dataset(
    dataset_name: str,
    output_dir: Path,
    split: str,
    max_samples: Optional[int] = None,
    generate_bev: bool = False
) -> List[Dict[str, Any]]:
    """Prepare data from Hugging Face dataset."""
    logger = logging.getLogger(__name__)
    
    if not HF_DATASETS_AVAILABLE:
        logger.error("Hugging Face datasets library not available")
        return []
    
    logger.info(f"Loading dataset from Hugging Face: {dataset_name}")
    
    try:
        # Try to load the dataset
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        
        metadata_list = []
        
        for idx, sample in enumerate(tqdm(dataset, desc=f"Processing {split}")):
            if max_samples and idx >= max_samples:
                break
            
            # Infer key mapping from first sample
            if idx == 0:
                key_mapping = {}
                for standard_key, possible_keys in DEFAULT_KEY_MAPPING.items():
                    found_key = find_key(sample.keys(), possible_keys)
                    if found_key:
                        key_mapping[standard_key] = found_key
            
            metadata = process_sample(
                sample, idx, split, output_dir,
                key_mapping, generate_bev
            )
            
            if metadata:
                metadata_list.append(metadata)
        
        logger.info(f"Processed {len(metadata_list)} samples from HF dataset")
        return metadata_list
        
    except Exception as e:
        logger.error(f"Error loading HF dataset: {e}")
        return []


def verify_prepared_data(output_dir: Path, split: str) -> bool:
    """Verify that prepared data is valid."""
    logger = logging.getLogger(__name__)
    
    index_file = output_dir / f'{split}_index.csv'
    if not index_file.exists():
        logger.error(f"Index file not found: {index_file}")
        return False
    
    df = pd.read_csv(index_file)
    logger.info(f"Split '{split}': {len(df)} samples")
    
    # Check a few random samples
    sample_indices = np.random.choice(len(df), min(10, len(df)), replace=False)
    
    for idx in sample_indices:
        row = df.iloc[idx]
        
        # Check image
        image_path = output_dir / row['path_to_image']
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return False
        
        # Check lidar
        lidar_path = output_dir / row['path_to_lidar']
        if not lidar_path.exists():
            logger.error(f"LiDAR not found: {lidar_path}")
            return False
        
        # Try to load
        try:
            img = Image.open(image_path)
            lidar = np.load(lidar_path)
            assert lidar.shape[1] == 4, f"Invalid lidar shape: {lidar.shape}"
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            return False
    
    logger.info(f"Verification passed for split '{split}'")
    return True


def main():
    parser = argparse.ArgumentParser(description='Prepare CARLA dataset for training')
    parser.add_argument('--data-dir', type=str, default='./CARLA_15GB/default',
                        help='Directory containing parquet files')
    parser.add_argument('--out', type=str, default='./data',
                        help='Output directory for processed data')
    parser.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'],
                        help='Data splits to process')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples per split (for testing)')
    parser.add_argument('--generate-bev', action='store_true',
                        help='Generate BEV projections')
    parser.add_argument('--image-quality', type=int, default=85,
                        help='JPEG quality (1-100)')
    parser.add_argument('--verify', action='store_true',
                        help='Only verify existing prepared data')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (verbose logging, 100 samples max)')
    parser.add_argument('--hf-dataset', type=str, default=None,
                        help='Hugging Face dataset name (alternative to local parquet)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.debug)
    
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Debug mode overrides
    if args.debug:
        args.max_samples = min(args.max_samples or 100, 100)
        logger.info("Debug mode: limiting to 100 samples per split")
    
    # Verify mode
    if args.verify:
        logger.info("Verification mode")
        all_valid = True
        for split in args.splits:
            if not verify_prepared_data(output_dir, split):
                all_valid = False
        
        if all_valid:
            logger.info("All splits verified successfully")
            return 0
        else:
            logger.error("Verification failed")
            return 1
    
    # Process each split
    for split in args.splits:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing split: {split}")
        logger.info(f"{'='*60}")
        
        if args.hf_dataset:
            metadata_list = prepare_from_hf_dataset(
                args.hf_dataset, output_dir, split,
                args.max_samples, args.generate_bev
            )
        else:
            data_dir = Path(args.data_dir)
            metadata_list = prepare_from_parquet(
                data_dir, output_dir, split,
                args.max_samples, args.generate_bev, args.debug
            )
        
        if not metadata_list:
            logger.warning(f"No samples processed for split '{split}'")
            continue
        
        # Save index CSV
        index_file = output_dir / f'{split}_index.csv'
        df = pd.DataFrame(metadata_list)
        df.to_csv(index_file, index=False)
        logger.info(f"Saved index: {index_file} ({len(df)} samples)")
        
        # Save summary stats
        stats = {
            'split': split,
            'num_samples': len(df),
            'steer_mean': float(df['steer'].mean()),
            'steer_std': float(df['steer'].std()),
            'throttle_mean': float(df['throttle'].mean()),
            'brake_mean': float(df['brake'].mean()),
            'speed_mean': float(df['speed_kmh'].mean()),
            'speed_max': float(df['speed_kmh'].max()),
        }
        
        stats_file = output_dir / f'{split}_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics: {stats}")
    
    logger.info("\n" + "="*60)
    logger.info("Data preparation complete!")
    logger.info("="*60)
    
    # Run verification
    logger.info("\nRunning verification...")
    all_valid = True
    for split in args.splits:
        if not verify_prepared_data(output_dir, split):
            all_valid = False
    
    if all_valid:
        logger.info("\n✓ All data verified successfully!")
        return 0
    else:
        logger.error("\n✗ Verification failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
