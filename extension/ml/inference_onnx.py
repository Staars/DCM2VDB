"""
ONNX-based inference for MedSAM2 on Windows/Linux

This module wraps the ONNX encoder/decoder models for efficient
inference on Windows and Linux platforms.
"""

import numpy as np
from pathlib import Path


class ONNXPredictor:
    """
    MedSAM2 predictor using ONNX backend
    
    Provides prompt-based segmentation for medical images using
    the MedSAM2 ONNX models.
    """
    
    def __init__(self, models_dir):
        """
        Initialize the ONNX predictor
        
        Args:
            models_dir: Path to models directory containing medsam2_onnx/
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not found. "
                "Make sure you're using the Windows/Linux extension bundle."
            )
        
        self.models_dir = Path(models_dir)
        self.encoder_path = self.models_dir / "medsam2_onnx" / "encoder.onnx"
        self.decoder_path = self.models_dir / "medsam2_onnx" / "decoder.onnx"
        
        if not self.encoder_path.exists():
            raise FileNotFoundError(
                f"ONNX encoder not found: {self.encoder_path}\n"
                "Make sure you're using the Windows/Linux extension bundle."
            )
        
        if not self.decoder_path.exists():
            raise FileNotFoundError(
                f"ONNX decoder not found: {self.decoder_path}\n"
                "Make sure you're using the Windows/Linux extension bundle."
            )
        
        # Load models
        print(f"Loading ONNX encoder from {self.encoder_path}")
        self.encoder_session = ort.InferenceSession(str(self.encoder_path))
        
        print(f"Loading ONNX decoder from {self.decoder_path}")
        self.decoder_session = ort.InferenceSession(str(self.decoder_path))
        
        print("✓ ONNX models loaded")
        
        # Cache for image embeddings (avoid re-encoding same image)
        self._cached_embeddings = None
        self._cached_image_hash = None
    
    def preprocess_image(self, image):
        """
        Preprocess image for model input
        
        Args:
            image: numpy array (H, W, 3) or (H, W) in range [0, 255]
            
        Returns:
            np.ndarray: Preprocessed image (1, 3, 1024, 1024)
        """
        # Convert to numpy if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Handle grayscale
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Ensure RGB
        if image.shape[-1] != 3:
            raise ValueError(f"Expected 3 channels, got {image.shape[-1]}")
        
        # Resize to 1024x1024
        h, w = image.shape[:2]
        if h != 1024 or w != 1024:
            from PIL import Image
            pil_img = Image.fromarray(image.astype(np.uint8))
            pil_img = pil_img.resize((1024, 1024), Image.BILINEAR)
            image = np.array(pil_img)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to CHW format
        image = np.transpose(image, (2, 0, 1))
        
        # Add batch dimension
        image = np.expand_dims(image, 0).astype(np.float32)
        
        return image
    
    def encode_image(self, image):
        """
        Encode image to embeddings
        
        Args:
            image: Preprocessed image (1, 3, 1024, 1024)
            
        Returns:
            dict: Image embeddings and features
        """
        # Check cache
        image_hash = hash(image.tobytes())
        if self._cached_image_hash == image_hash:
            return self._cached_embeddings
        
        # Run encoder
        outputs = self.encoder_session.run(None, {'image': image})
        
        embeddings = {
            'image_embeddings': outputs[0],
            'backbone_fpn_0': outputs[1],
            'backbone_fpn_1': outputs[2],
        }
        
        # Cache results
        self._cached_embeddings = embeddings
        self._cached_image_hash = image_hash
        
        return embeddings
    
    def decode_masks(self, embeddings, point_coords, point_labels):
        """
        Decode masks from embeddings and prompts
        
        Args:
            embeddings: Dict of image embeddings
            point_coords: Array of point coordinates (N, 2) in [0, 1024]
            point_labels: Array of point labels (N,) - 1 for positive, 0 for negative
            
        Returns:
            tuple: (masks, iou_predictions)
        """
        # Prepare inputs
        coords = np.expand_dims(point_coords, 0).astype(np.float32)  # (1, N, 2)
        labels = np.expand_dims(point_labels, 0).astype(np.float32)  # (1, N)
        
        # Run decoder
        outputs = self.decoder_session.run(
            None,
            {
                'image_embeddings': embeddings['image_embeddings'],
                'backbone_fpn_0': embeddings['backbone_fpn_0'],
                'backbone_fpn_1': embeddings['backbone_fpn_1'],
                'point_coords': coords,
                'point_labels': labels,
            }
        )
        
        masks = outputs[0]
        iou_predictions = outputs[1]
        
        return masks, iou_predictions
    
    def segment(self, image, points, labels=None):
        """
        Segment image with point prompts
        
        Args:
            image: Input image (H, W, 3) or (H, W)
            points: List of (x, y) coordinates or array (N, 2)
            labels: List of labels (1=positive, 0=negative) or None (all positive)
            
        Returns:
            dict: Segmentation results
                'mask': Binary mask (H, W)
                'iou': IoU prediction score
        """
        # Convert points to array
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        
        # Default labels to all positive
        if labels is None:
            labels = np.ones(len(points))
        elif not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        
        # Preprocess image
        preprocessed = self.preprocess_image(image)
        
        # Encode image
        embeddings = self.encode_image(preprocessed)
        
        # Decode masks
        masks, iou_predictions = self.decode_masks(
            embeddings,
            points,
            labels
        )
        
        # Extract results
        mask = masks[0, 0]  # (256, 256)
        iou = float(iou_predictions[0, 0])
        
        # Debug: log mask statistics
        print(f"Mask stats - min: {mask.min():.4f}, max: {mask.max():.4f}, mean: {mask.mean():.4f}")
        
        # Threshold mask at 0.0 (model outputs logits, positive = inside mask)
        mask = (mask > 0.0).astype(np.float32)
        
        return {
            'mask': mask,
            'iou': iou,
        }
