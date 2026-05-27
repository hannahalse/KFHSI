#!/usr/bin/env python3
"""
Preprocess a calibration image the same way scan frames are saved:
- crop in Y
- bin in X

This is useful when the calibration image was captured at full camera
resolution, but the scan pipeline saves cropped+binned frames.
"""

import argparse
import os

import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_PATH = os.path.join(BASE_DIR, "ar_200ms.png")
DEFAULT_ROI_TOP = 248
DEFAULT_ROI_BOTTOM = 803
DEFAULT_BINNING_FACTOR = 2


def load_gray_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


def crop_roi(image, roi_top, roi_bottom):
    if roi_top < 0 or roi_bottom > image.shape[0] or roi_bottom <= roi_top:
        raise ValueError(
            f"Invalid ROI rows [{roi_top}:{roi_bottom}] for image with height {image.shape[0]}"
        )
    return image[roi_top:roi_bottom, :]


def bin_image_x(image, factor):
    if factor <= 0:
        raise ValueError("binning factor must be > 0")

    height, width = image.shape
    new_width = width // factor
    if new_width == 0:
        raise ValueError(
            f"Binning factor {factor} is too large for width {width}"
        )

    trimmed = image[:, : new_width * factor]
    binned = trimmed.reshape(height, new_width, factor).mean(axis=2)
    return binned.astype(np.uint8)


def preprocess_image(image, roi_top, roi_bottom, binning_factor, flip_x=False):
    processed = image
    if flip_x:
        processed = np.fliplr(processed)
    processed = crop_roi(processed, roi_top, roi_bottom)
    processed = bin_image_x(processed, binning_factor)
    return processed


def build_default_output_path(input_path, roi_top, roi_bottom, binning_factor, flip_x):
    stem, ext = os.path.splitext(os.path.basename(input_path))
    suffix = f"_roi{roi_top}-{roi_bottom}_bin{binning_factor}"
    if flip_x:
        suffix += "_flipped"
    return os.path.join(BASE_DIR, f"{stem}{suffix}{ext}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop and bin a calibration image to match saved scan frames."
    )
    parser.add_argument(
        "--input-path",
        default=DEFAULT_INPUT_PATH,
        help="Path to the input calibration image.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Where to save the processed image. Defaults to a generated filename in calibration/.",
    )
    parser.add_argument(
        "--roi-top",
        type=int,
        default=DEFAULT_ROI_TOP,
        help="Top row of the Y ROI (inclusive).",
    )
    parser.add_argument(
        "--roi-bottom",
        type=int,
        default=DEFAULT_ROI_BOTTOM,
        help="Bottom row of the Y ROI (exclusive).",
    )
    parser.add_argument(
        "--binning-factor",
        type=int,
        default=DEFAULT_BINNING_FACTOR,
        help="Binning factor along X (spectral axis).",
    )
    parser.add_argument(
        "--flip-x",
        action="store_true",
        help="Flip the image horizontally before crop/binning.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = os.path.abspath(args.input_path)
    output_path = args.output_path
    if output_path is None:
        output_path = build_default_output_path(
            input_path,
            args.roi_top,
            args.roi_bottom,
            args.binning_factor,
            args.flip_x,
        )
    output_path = os.path.abspath(output_path)

    image = load_gray_image(input_path)
    processed = preprocess_image(
        image,
        roi_top=args.roi_top,
        roi_bottom=args.roi_bottom,
        binning_factor=args.binning_factor,
        flip_x=args.flip_x,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, processed)

    print(f"Input image:     {input_path}")
    print(f"Output image:    {output_path}")
    print(f"Original shape:  {image.shape}")
    print(f"Processed shape: {processed.shape}")
    print(
        "Settings:"
        f" roi_top={args.roi_top},"
        f" roi_bottom={args.roi_bottom},"
        f" binning_factor={args.binning_factor},"
        f" flip_x={args.flip_x}"
    )


if __name__ == "__main__":
    main()
