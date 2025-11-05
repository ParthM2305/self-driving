"""
Utility functions for the self-driving model project.
Includes: logging, checkpoint management, seed control, device detection.
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch


def setup_logging(log_file: Optional[str] = None, level=logging.INFO):
    """Configure logging to console and optionally to file."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make cudnn deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logging.info(f"Random seed set to {seed}")


def get_device(device_str: str = 'auto') -> torch.device:
    """
    Get the appropriate device for training.
    
    Args:
        device_str: 'auto', 'cuda', 'cpu', or specific device like 'cuda:0'
    
    Returns:
        torch.device object
    """
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    if device.type == 'cuda':
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logging.info("Using CPU (training will be slower)")
    
    return device


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    is_best: bool = False
):
    """
    Save model checkpoint with all training state.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer state
        scheduler: Learning rate scheduler
        epoch: Current epoch number
        metrics: Dictionary of validation metrics
        save_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Handle DataParallel models
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }
    
    torch.save(checkpoint, save_path)
    logging.info(f"Checkpoint saved: {save_path}")
    
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)
        logging.info(f"Best checkpoint saved: {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    Load checkpoint and restore training state.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        device: Device to load tensors to
    
    Returns:
        Dictionary with epoch and metrics
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load with weights_only=False for compatibility with numpy scalars
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle DataParallel models
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logging.info(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
    
    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint.get('metrics', {})
    }


def save_metrics(metrics: Dict[str, Any], save_path: str):
    """Save metrics dictionary to JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logging.info(f"Metrics saved: {save_path}")


def load_metrics(metrics_path: str) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_auto_batch_size(device: torch.device, default_gpu: int = 64, default_cpu: int = 8) -> int:
    """Get appropriate batch size based on device."""
    if device.type == 'cuda':
        # Try to estimate based on GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb < 6:
            return 16
        elif gpu_memory_gb < 12:
            return 32
        else:
            return default_gpu
    else:
        return default_cpu


def get_auto_num_workers(device: torch.device) -> int:
    """Get appropriate number of DataLoader workers."""
    if device.type == 'cuda':
        return min(8, os.cpu_count() or 4)
    else:
        return 2  # Lower for CPU to avoid overhead


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def create_run_dir(base_dir: str, run_name: Optional[str] = None) -> str:
    """
    Create a unique directory for this training run.
    
    Args:
        base_dir: Base directory for all runs
        run_name: Optional custom run name
    
    Returns:
        Path to created run directory
    """
    from datetime import datetime
    
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
    
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'visualizations'), exist_ok=True)
    
    return run_dir


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration dictionary to YAML or JSON."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if save_path.endswith('.yaml') or save_path.endswith('.yml'):
        import yaml
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    logging.info(f"Config saved: {save_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON."""
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    return config
