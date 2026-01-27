"""Test and benchmark compute backend performance"""

import bpy
import time
import numpy as np
from bpy.types import Operator
from .compute_backend import xp, backend_name, to_numpy, from_numpy, get_backend_info
from .utils import SimpleLogger

log = SimpleLogger()


class DICOM_OT_test_compute_backend(Operator):
    """Test GPU compute backend and show performance comparison"""
    bl_idname = "dicom.test_compute_backend"
    bl_label = "Test Compute Backend"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        log.info("=" * 60)
        log.info("COMPUTE BACKEND TEST")
        log.info("=" * 60)
        
        # Show backend info
        info = get_backend_info()
        log.info(f"Backend: {info['name']}")
        log.info(f"GPU Accelerated: {info['gpu_accelerated']}")
        log.info(f"Device: {info['device']}")
        
        if 'memory_total' in info:
            log.info(f"GPU Memory: {info['memory_free']} / {info['memory_total']}")
        
        log.info("-" * 60)
        
        # Run benchmarks
        self._benchmark_basic_ops()
        self._benchmark_volume_ops()
        self._benchmark_large_volume()
        
        log.info("=" * 60)
        self.report({'INFO'}, f"Backend test complete. Using: {backend_name}")
        
        return {'FINISHED'}
    
    def _benchmark_basic_ops(self):
        """Benchmark basic array operations"""
        log.info("Basic Operations Benchmark:")
        
        size = 1000
        iterations = 100
        
        # NumPy baseline
        np_data = np.random.rand(size, size).astype(np.float32)
        
        start = time.time()
        for _ in range(iterations):
            result = np.sum(np_data * np_data + 1.0)
        numpy_time = time.time() - start
        
        log.info(f"  NumPy ({size}x{size}, {iterations} iterations): {numpy_time:.3f}s")
        
        # Backend test
        backend_data = from_numpy(np_data)
        
        start = time.time()
        for _ in range(iterations):
            result = xp.sum(backend_data * backend_data + 1.0)
            if backend_name != 'numpy':
                # Force synchronization for accurate timing
                _ = to_numpy(result)
        backend_time = time.time() - start
        
        log.info(f"  {backend_name.upper()} ({size}x{size}, {iterations} iterations): {backend_time:.3f}s")
        
        if backend_name != 'numpy':
            speedup = numpy_time / backend_time
            log.info(f"  Speedup: {speedup:.2f}x")
    
    def _benchmark_volume_ops(self):
        """Benchmark volume-like operations"""
        log.info("Volume Operations Benchmark:")
        
        # Simulate a small CT volume
        volume_shape = (100, 256, 256)  # Z, Y, X
        
        # NumPy baseline
        np_volume = np.random.randint(-1000, 3000, volume_shape, dtype=np.int16)
        
        start = time.time()
        mask = (np_volume > 200) & (np_volume < 1500)
        bone_voxels = np.sum(mask)
        mean_hu = np.mean(np_volume[mask])
        numpy_time = time.time() - start
        
        log.info(f"  NumPy (threshold + stats): {numpy_time:.3f}s")
        log.info(f"    Bone voxels: {bone_voxels}, Mean HU: {mean_hu:.1f}")
        
        # Backend test
        backend_volume = from_numpy(np_volume)
        
        start = time.time()
        mask_gpu = (backend_volume > 200) & (backend_volume < 1500)
        bone_voxels_gpu = xp.sum(mask_gpu)
        
        # MLX doesn't support boolean indexing yet, use where instead
        if backend_name == 'mlx':
            # Use where to get values where mask is true
            masked_values = xp.where(mask_gpu, backend_volume, 0)
            mean_hu_gpu = xp.sum(masked_values) / xp.maximum(bone_voxels_gpu, 1)
        else:
            mean_hu_gpu = xp.mean(backend_volume[mask_gpu])
        
        # Convert results back
        bone_voxels_result = int(to_numpy(bone_voxels_gpu))
        mean_hu_result = float(to_numpy(mean_hu_gpu))
        backend_time = time.time() - start
        
        log.info(f"  {backend_name.upper()} (threshold + stats): {backend_time:.3f}s")
        log.info(f"    Bone voxels: {bone_voxels_result}, Mean HU: {mean_hu_result:.1f}")
        
        if backend_name != 'numpy':
            speedup = numpy_time / backend_time
            log.info(f"  Speedup: {speedup:.2f}x")
    
    def _benchmark_large_volume(self):
        """Benchmark with realistic large CT volume"""
        log.info("Large Volume Benchmark (512x512x300):")
        
        # Realistic CT scan size
        volume_shape = (300, 512, 512)  # ~300 MB
        
        log.info(f"  Volume size: {volume_shape[0]}x{volume_shape[1]}x{volume_shape[2]} = {np.prod(volume_shape)/1e6:.1f}M voxels")
        log.info(f"  Memory: ~{np.prod(volume_shape)*2/1024**2:.0f} MB")
        
        # NumPy baseline
        np_volume = np.random.randint(-1000, 3000, volume_shape, dtype=np.int16)
        
        start = time.time()
        # Multi-step processing: threshold, count, statistics
        mask = (np_volume > 200) & (np_volume < 1500)
        bone_voxels = np.sum(mask)
        mean_hu = np.mean(np_volume[mask])
        std_hu = np.std(np_volume[mask])
        numpy_time = time.time() - start
        
        log.info(f"  NumPy: {numpy_time:.3f}s")
        log.info(f"    Bone voxels: {bone_voxels}, Mean: {mean_hu:.1f}, Std: {std_hu:.1f}")
        
        # Backend test with lazy evaluation
        start = time.time()
        backend_volume = from_numpy(np_volume)
        
        # Chain operations (lazy evaluation in MLX)
        mask_gpu = (backend_volume > 200) & (backend_volume < 1500)
        bone_voxels_gpu = xp.sum(mask_gpu)
        
        if backend_name == 'mlx':
            # MLX: Use where for masked operations
            masked_values = xp.where(mask_gpu, backend_volume, 0)
            count_gpu = xp.sum(mask_gpu)
            sum_gpu = xp.sum(masked_values)
            mean_hu_gpu = sum_gpu / xp.maximum(count_gpu, 1)
            
            # Std calculation - need to recalculate with correct mean
            diff = xp.where(mask_gpu, backend_volume - mean_hu_gpu, 0)
            variance = xp.sum(diff * diff) / xp.maximum(count_gpu, 1)
            std_hu_gpu = xp.sqrt(variance)
            
            # Force evaluation of all operations at once
            import mlx.core as mx
            mx.eval(count_gpu, mean_hu_gpu, std_hu_gpu)
            
            bone_voxels_gpu = count_gpu
        else:
            mean_hu_gpu = xp.mean(backend_volume[mask_gpu])
            std_hu_gpu = xp.std(backend_volume[mask_gpu])
        
        # Convert results
        bone_voxels_result = int(to_numpy(bone_voxels_gpu))
        mean_hu_result = float(to_numpy(mean_hu_gpu))
        std_hu_result = float(to_numpy(std_hu_gpu))
        backend_time = time.time() - start
        
        log.info(f"  {backend_name.upper()}: {backend_time:.3f}s")
        log.info(f"    Bone voxels: {bone_voxels_result}, Mean: {mean_hu_result:.1f}, Std: {std_hu_result:.1f}")
        
        if backend_name != 'numpy':
            speedup = numpy_time / backend_time
            log.info(f"  Speedup: {speedup:.2f}x")
            
            if speedup > 1.0:
                log.info(f"  ✓ GPU acceleration effective at this scale!")
            else:
                log.info(f"  ⚠ GPU overhead still dominates (try larger volumes)")


def register():
    bpy.utils.register_class(DICOM_OT_test_compute_backend)


def unregister():
    bpy.utils.unregister_class(DICOM_OT_test_compute_backend)
