"""
Enhanced data augmentation module for wildlife image classification.

This module provides both standard augmentations and quality-simulating augmentations
to help models generalize better to real-world image variations, including poor
lighting, blur, noise, and compression artifacts.
"""

import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from typing import Tuple


class RobustAugmentation:
    """
    Comprehensive augmentation pipeline that simulates real-world image quality variations.

    This class combines standard augmentations with quality-simulating augmentations
    to help models generalize better to the full range of image qualities in wildlife
    camera trap data.
    """

    def __init__(
        self,
        # Standard augmentation parameters
        rotation_range: int = 15,
        horizontal_flip_p: float = 0.5,
        vertical_flip_p: float = 0.0,
        brightness_range: tuple[float, float] = (0.8, 1.2),
        contrast_range: tuple[float, float] = (0.8, 1.2),
        saturation_range: tuple[float, float] = (0.8, 1.2),
        hue_range: tuple[float, float] = (-0.1, 0.1),
        # Quality-simulating augmentation parameters
        blur_p: float = 0.3,
        blur_kernel_range: tuple[int, int] = (3, 7),
        noise_p: float = 0.2,
        noise_std_range: tuple[float, float] = (0.01, 0.05),
        compression_p: float = 0.1,
        compression_quality_range: tuple[int, int] = (30, 90),
        jpeg_artifacts_p: float = 0.15,
        motion_blur_p: float = 0.1,
        motion_blur_kernel_range: tuple[int, int] = (5, 15),
        # Color jitter for lighting variations
        color_jitter_p: float = 0.4,
        # Enable/disable specific augmentations
        enable_standard: bool = True,
        enable_quality_simulation: bool = True,
        # Random seed for reproducibility
        seed: int | None = None,
    ):
        """
        Initialize the robust augmentation pipeline.

        Args:
            rotation_range: Maximum rotation angle in degrees
            horizontal_flip_p: Probability of horizontal flip
            vertical_flip_p: Probability of vertical flip
            brightness_range: Range for brightness adjustment (multiplier)
            contrast_range: Range for contrast adjustment (multiplier)
            saturation_range: Range for saturation adjustment (multiplier)
            hue_range: Range for hue adjustment (shift in radians)
            blur_p: Probability of applying Gaussian blur
            blur_kernel_range: Range for blur kernel size
            noise_p: Probability of adding Gaussian noise
            noise_std_range: Range for noise standard deviation
            compression_p: Probability of JPEG compression simulation
            compression_quality_range: Range for JPEG quality (0-100)
            jpeg_artifacts_p: Probability of JPEG artifact simulation
            motion_blur_p: Probability of motion blur simulation
            motion_blur_kernel_range: Range for motion blur kernel size
            mixup_p: Probability of mixup augmentation
            mixup_alpha: Mixup alpha parameter
            cutmix_p: Probability of cutmix augmentation
            cutmix_alpha: Cutmix alpha parameter
            random_erase_p: Probability of random erasing
            random_erase_scale: Range for random erase scale
            random_erase_ratio: Range for random erase aspect ratio
            color_jitter_p: Probability of color jitter
            enable_standard: Enable standard augmentations
            enable_quality_simulation: Enable quality-simulating augmentations
            enable_advanced: Enable advanced augmentations (mixup, cutmix)
            seed: Random seed for reproducibility
        """
        self.rotation_range = rotation_range
        self.horizontal_flip_p = horizontal_flip_p
        self.vertical_flip_p = vertical_flip_p
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range

        self.blur_p = blur_p
        self.blur_kernel_range = blur_kernel_range
        self.noise_p = noise_p
        self.noise_std_range = noise_std_range
        self.compression_p = compression_p
        self.compression_quality_range = compression_quality_range
        self.jpeg_artifacts_p = jpeg_artifacts_p
        self.motion_blur_p = motion_blur_p
        self.motion_blur_kernel_range = motion_blur_kernel_range

        self.color_jitter_p = color_jitter_p

        self.enable_standard = enable_standard
        self.enable_quality_simulation = enable_quality_simulation

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def __call__(self, image: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Apply augmentations to the image.

        Args:
            image: Input image (numpy array or torch tensor)

        Returns:
            Augmented image
        """
        # Convert to numpy if needed
        if isinstance(image, torch.Tensor):
            is_tensor = True
            if image.dim() == 4:  # Batch dimension
                image = image.squeeze(0)
            image_np = image.permute(1, 2, 0).numpy()
        else:
            is_tensor = False
            image_np = image.copy()

        # Apply standard augmentations
        if self.enable_standard:
            image_np = self._apply_standard_augmentations(image_np)

        # Apply quality-simulating augmentations
        if self.enable_quality_simulation:
            image_np = self._apply_quality_simulations(image_np)

        # Convert back to tensor if needed
        if is_tensor:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
            return image_tensor
        else:
            return image_np

    def _apply_standard_augmentations(self, image: np.ndarray) -> np.ndarray:
        """Apply standard geometric and color augmentations."""
        # Random rotation
        if random.random() < 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image = self._rotate_image(image, angle)

        # Random flips
        if random.random() < self.horizontal_flip_p:
            image = cv2.flip(image, 1)  # Horizontal flip

        if random.random() < self.vertical_flip_p:
            image = cv2.flip(image, 0)  # Vertical flip

        # Color jitter
        if random.random() < self.color_jitter_p:
            image = self._color_jitter(image)

        return image

    def _apply_quality_simulations(self, image: np.ndarray) -> np.ndarray:
        """Apply quality-simulating augmentations."""
        # Gaussian blur
        if random.random() < self.blur_p:
            kernel_size = random.randint(*self.blur_kernel_range)
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size
            sigma = random.uniform(0.5, 2.0)
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

        # Motion blur
        if random.random() < self.motion_blur_p:
            kernel_size = random.randint(*self.motion_blur_kernel_range)
            if kernel_size % 2 == 0:
                kernel_size += 1
            angle = random.uniform(0, 180)
            image = self._motion_blur(image, kernel_size, angle)

        # Gaussian noise
        if random.random() < self.noise_p:
            std = random.uniform(*self.noise_std_range)
            noise = np.random.normal(0, std, image.shape).astype(np.float32)
            image = image.astype(np.float32) + noise
            image = np.clip(image, 0, 255).astype(np.uint8)

        # JPEG compression simulation
        if random.random() < self.compression_p:
            quality = random.randint(*self.compression_quality_range)
            image = self._simulate_jpeg_compression(image, quality)

        # JPEG artifacts simulation
        if random.random() < self.jpeg_artifacts_p:
            image = self._simulate_jpeg_artifacts(image)

        return image

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle."""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (width, height))

    def _color_jitter(self, image: np.ndarray) -> np.ndarray:
        """Apply color jitter transformations."""
        # Convert to float for calculations
        image_float = image.astype(np.float32) / 255.0

        # Brightness
        brightness_factor = random.uniform(*self.brightness_range)
        image_float = image_float * brightness_factor

        # Contrast
        contrast_factor = random.uniform(*self.contrast_range)
        mean = np.mean(image_float)
        image_float = (image_float - mean) * contrast_factor + mean

        # Convert to HSV for saturation and hue
        if len(image_float.shape) == 3:
            hsv = cv2.cvtColor(image_float, cv2.COLOR_RGB2HSV)

            # Saturation
            saturation_factor = random.uniform(*self.saturation_range)
            hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor

            # Hue
            hue_shift = random.uniform(*self.hue_range)
            hsv[:, :, 0] = hsv[:, :, 0] + hue_shift

            # Convert back to RGB
            image_float = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Clip and convert back to uint8
        image_float = np.clip(image_float, 0, 1)
        return (image_float * 255).astype(np.uint8)

    def _motion_blur(self, image: np.ndarray, kernel_size: int, angle: float) -> np.ndarray:
        """Apply motion blur with given kernel size and angle."""
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2

        # Calculate direction
        angle_rad = np.radians(angle)
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)

        # Create line in kernel
        for i in range(kernel_size):
            x = int(center + (i - center) * dx)
            y = int(center + (i - center) * dy)
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1

        # Normalize kernel
        kernel = kernel / np.sum(kernel)

        # Apply convolution
        return cv2.filter2D(image, -1, kernel)

    def _simulate_jpeg_compression(self, image: np.ndarray, quality: int) -> np.ndarray:
        """Simulate JPEG compression artifacts."""
        # Encode and decode with specified quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode(".jpg", image, encode_param)
        decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

        # Handle grayscale images
        if len(image.shape) == 2:
            decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2GRAY)

        return decoded_img

    def _simulate_jpeg_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Simulate JPEG compression artifacts more realistically."""
        # Convert to YUV color space
        if len(image.shape) == 3:
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else:
            yuv = image.copy()

        # Add quantization noise to chrominance channels
        if len(image.shape) == 3:
            # Add more noise to U and V channels (chrominance)
            noise_u = np.random.normal(0, 2, yuv.shape[:2]).astype(np.int16)
            noise_v = np.random.normal(0, 2, yuv.shape[:2]).astype(np.int16)

            yuv[:, :, 1] = np.clip(yuv[:, :, 1].astype(np.int16) + noise_u, 0, 255).astype(
                np.uint8
            )
            yuv[:, :, 2] = np.clip(yuv[:, :, 2].astype(np.int16) + noise_v, 0, 255).astype(
                np.uint8
            )

        # Convert back to RGB
        if len(image.shape) == 3:
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        else:
            return yuv


class WildlifeAugmentationDataset(torch.utils.data.Dataset):
    """
    Custom dataset that applies robust augmentations during training.

    This dataset wraps the original data and applies augmentations on-the-fly,
    allowing for different augmentations for each epoch.
    """

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        augmentation: RobustAugmentation | None = None,
        is_training: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            images: Tensor of images (N, C, H, W)
            labels: Tensor of labels
            augmentation: Augmentation pipeline to apply
            is_training: Whether to apply augmentations (only for training)
        """
        self.images = images
        self.labels = labels
        self.augmentation = augmentation
        self.is_training = is_training

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.is_training and self.augmentation is not None:
            # Apply augmentations
            image = self.augmentation(image)

        return image, label


class ExpandedAugmentationDataset(torch.utils.data.Dataset):
    """
    Dataset that expands the training set by including both original and augmented images.

    This creates a 2x larger training set where each original image is paired with
    an augmented version, effectively doubling the dataset size.
    """

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        augmentation: RobustAugmentation | None = None,
        is_training: bool = True,
    ):
        """
        Initialize the expanded dataset.

        Args:
            images: Tensor of images (N, C, H, W)
            labels: Tensor of labels
            augmentation: Augmentation pipeline to apply
            is_training: Whether to include augmented versions (only for training)
        """
        self.images = images
        self.labels = labels
        self.augmentation = augmentation
        self.is_training = is_training
        # Calculate dataset size: original + augmented (if training)
        self.original_size = len(images)
        self.expanded_size = (
            self.original_size * 2 if is_training and augmentation else self.original_size
        )

    def __len__(self):
        return self.expanded_size

    def __getitem__(self, idx):
        # Determine if this is an original or augmented image
        is_augmented = idx >= self.original_size
        if is_augmented:
            # Get the corresponding original image
            original_idx = idx - self.original_size
            image = self.images[original_idx]
            label = self.labels[original_idx]
            # Apply augmentation
            if self.augmentation is not None:
                image = self.augmentation(image)
        else:
            # Return original image
            image = self.images[idx]
            label = self.labels[idx]
        return image, label


def create_augmentation_pipeline(
    augmentation_strength: str = "medium",
    enable_quality_simulation: bool = True,
    seed: int | None = None,
) -> RobustAugmentation:
    """
    Create an augmentation pipeline with predefined configurations.

    Args:
        augmentation_strength: Strength of augmentations ("light", "medium", "strong")
        enable_quality_simulation: Enable quality-simulating augmentations
        enable_advanced: Enable advanced augmentations (mixup, cutmix)
        seed: Random seed for reproducibility

    Returns:
        Configured RobustAugmentation instance
    """

    if augmentation_strength == "light":
        return RobustAugmentation(
            rotation_range=10,
            horizontal_flip_p=0.5,
            brightness_range=(0.9, 1.1),
            contrast_range=(0.9, 1.1),
            blur_p=0.2,
            noise_p=0.1,
            compression_p=0.05,
            enable_quality_simulation=enable_quality_simulation,
            seed=seed,
        )
    elif augmentation_strength == "medium":
        return RobustAugmentation(
            rotation_range=15,
            horizontal_flip_p=0.5,
            brightness_range=(0.8, 1.2),
            contrast_range=(0.8, 1.2),
            saturation_range=(0.8, 1.2),
            hue_range=(-0.1, 0.1),
            blur_p=0.3,
            noise_p=0.2,
            compression_p=0.1,
            jpeg_artifacts_p=0.15,
            motion_blur_p=0.1,
            enable_quality_simulation=enable_quality_simulation,
            seed=seed,
        )
    elif augmentation_strength == "strong":
        return RobustAugmentation(
            rotation_range=20,
            horizontal_flip_p=0.5,
            vertical_flip_p=0.1,
            brightness_range=(0.7, 1.3),
            contrast_range=(0.7, 1.3),
            saturation_range=(0.7, 1.3),
            hue_range=(-0.15, 0.15),
            blur_p=0.4,
            noise_p=0.3,
            compression_p=0.15,
            jpeg_artifacts_p=0.2,
            motion_blur_p=0.15,
            enable_quality_simulation=enable_quality_simulation,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown augmentation strength: {augmentation_strength}")


def apply_batch_augmentations(
    images: torch.Tensor,
    augmentation: RobustAugmentation,
) -> torch.Tensor:
    """
    Apply augmentations to a batch of images.

    Args:
        images: Batch of images (B, C, H, W)
        augmentation: Augmentation pipeline

    Returns:
        Augmented images
    """
    batch_size = images.shape[0]

    # Apply individual augmentations to each image
    augmented_images = []
    for i in range(batch_size):
        aug_image = augmentation(images[i])
        augmented_images.append(aug_image)

    return torch.stack(augmented_images)
