"""GPU-accelerated compute backend and filters"""

from .backend import xp, backend_name, to_numpy, from_numpy, get_backend_info
from .utils import threshold_volume_gpu, calculate_volume_statistics_gpu
from .test import DICOM_OT_test_compute_backend
from .filters import (
    gaussian_filter_gpu, gaussian_filter_3d_gpu,
    percentile_filter_gpu, median_filter_gpu,
    denoise_slice_gpu, denoise_volume_batch_gpu,
)
