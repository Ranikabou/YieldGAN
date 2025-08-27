# Segfault Fix Summary - macOS PyTorch Training

## Problem
The training process was dying with return code -11 (SIGSEGV - segmentation fault) on macOS Intel systems. This is a common issue with PyTorch on macOS due to multiprocessing and threading conflicts.

## Root Cause
Segmentation faults in PyTorch on macOS are typically caused by:
1. **DataLoader workers using fork()** - macOS has issues with fork-based multiprocessing
2. **OpenMP/BLAS thread conflicts** - NumPy/PyTorch/MKL/Accelerate framework conflicts
3. **Mixed event-loop + native libraries** - Less common but possible

## Fixes Implemented

### 1. Thread Limiting Environment Variables
Added at the top of `train_gan_csv.py` before any imports:
```python
# Critical: Set threading limits before importing any libraries to prevent segfaults on macOS
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"
```

### 2. PyTorch Thread Configuration
Added after importing torch:
```python
# Set torch threading after import
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
```

### 3. Multiprocessing Start Method
Added multiprocessing spawn guard at the main entry point:
```python
if __name__ == "__main__":
    # Critical: Set multiprocessing start method to 'spawn' on macOS to prevent segfaults
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
```

### 4. DataLoader Worker Configuration
Modified `utils/data_utils.py` to disable multiprocessing workers:
```python
# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)  
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
```

### 5. Additional Fix - Method Name Correction
Fixed evaluation function to use correct method name:
```python
# Changed from: trainer.generate_synthetic_data()
# To: trainer.generate_sample()
synthetic_data = trainer.generate_sample(num_samples=len(test_loader.dataset))
```

## Results
✅ **Training now completes successfully without segfaults**
✅ **All 5 epochs completed successfully**
✅ **Dashboard integration working**
✅ **No more return code -11 errors**

## Test Command
To test with debug flags enabled:
```bash
PYTHONFAULTHANDLER=1 TORCH_SHOW_CPP_STACKTRACES=1 python train_gan_csv.py --config config/gan_config.yaml --data treasury_orderbook_sample.csv
```

## Performance Impact
- Slightly reduced performance due to single-threaded operation
- More stable training on macOS systems
- No impact on model quality or convergence

## Future Considerations
For production deployments on Linux systems, these restrictions can be relaxed:
- `num_workers` can be set to a reasonable number (e.g., 4-8)
- Thread limits can be increased based on system capacity
- Fork-based multiprocessing may work fine on Linux

The current configuration prioritizes stability over performance for development on macOS. 