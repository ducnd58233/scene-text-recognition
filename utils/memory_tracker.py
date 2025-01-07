from typing import Optional
import torch
import gc

class MemoryTracker:
    """Track and manage memory usage during training."""
    @staticmethod
    def get_memory_stats():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            return allocated, reserved
        return 0, 0

    @staticmethod
    def clear_memory(model: Optional[torch.nn.Module] = None):
        """Thoroughly clear GPU and CPU memory."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Clear Python's garbage collector
            gc.collect()
            
            # Release CPU memory if possible
            if model and 'cuda' in str(next(model.parameters()).device):
                with torch.cuda.device('cuda'):
                    torch.cuda.empty_cache()
                    gc.collect()
        except Exception as e:
            print(f"Memory clearing error: {str(e)}")

def safe_to_device(tensor, device, non_blocking=True):
    """Safely move tensor to device with error handling."""
    try:
        return tensor.to(device, non_blocking=non_blocking)
    except RuntimeError as e:
        MemoryTracker.clear_memory()
        try:
            return tensor.to(device, non_blocking=False)
        except RuntimeError:
            raise RuntimeError(f"Failed to move tensor to device even after memory cleanup: {str(e)}")
