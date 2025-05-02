import os
import re
import io
import time
import gc
import tempfile
import zipfile
import shutil
import socket
import traceback
import base64
import math
import numpy as np
import pandas as pd
import requests
import urllib.parse
import streamlit as st
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from io import BytesIO
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

# Try to import PDF-related modules, falling back to None if they're not available
pdf_support = False
pdf_image_support = False
poppler_available = False
pdf_conversion_errors = []

try:
    from pypdf import PdfReader  # New PyPDF2 is called pypdf
    pdf_support = True
    pdf_reader_name = "pypdf"
except ImportError:
    try:
        import PyPDF2  # Fall back to old name
        pdf_support = True
        pdf_reader_name = "PyPDF2"
    except ImportError:
        pdf_reader_name = None
        pdf_conversion_errors.append("PDF libraries not found: Install 'pypdf' for PDF reading capabilities")

# Try local PDF conversion first
try:
    from pdf2image import convert_from_bytes
    pdf_image_support = True
    
    # Check if poppler is actually available by attempting a simple conversion
    try:
        # Create a minimal 1x1 PDF in memory for testing
        from io import BytesIO
        import PIL.Image
        
        # Very small valid PDF content for testing
        test_pdf_content = b'%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 1 1]>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n149\n%EOF\n'
        test_pdf = BytesIO(test_pdf_content)
        
        try:
            # Try to convert the test PDF
            test_result = convert_from_bytes(test_pdf.getvalue(), dpi=72, first_page=1, last_page=1)
            if test_result and len(test_result) > 0:
                poppler_available = True
                print("Poppler test successful: PDF to image conversion is available")
            else:
                pdf_conversion_errors.append("PDF test conversion produced no images")
        except Exception as e:
            error_msg = str(e)
            if "poppler" in error_msg.lower():
                pdf_conversion_errors.append(f"Poppler not properly installed: {error_msg}")
            else:
                pdf_conversion_errors.append(f"PDF conversion test failed: {error_msg}")
    except Exception as e:
        pdf_conversion_errors.append(f"Error during PDF conversion test: {str(e)}")
except ImportError:
    pdf_conversion_errors.append("pdf2image module not found: Install 'pdf2image' for PDF to image conversion")

# Define a cloud-based PDF conversion function using PDF.co Web API
# This is a fallback when local conversion is not available
def convert_pdf_with_cloud_api(pdf_bytes):
    """
    Convert PDF to image using PDF.co Web API when Poppler is not available
    """
    try:
        # PDF.co offers a free tier with limited API calls
        # You should replace this with your own API key
        PDFCO_API_KEY = st.secrets.get("PDFCO_API_KEY", "")
        
        if not PDFCO_API_KEY:
            # Fallback to a temporary solution for the demo
            # In production, ALWAYS use proper API keys stored in st.secrets
            PDFCO_API_KEY = "demo"
            print("WARNING: Using demo API key. For production, use st.secrets to store your API key.")
        
        # Convert PDF to base64 for API transfer
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
        
        # Prepare API request
        url = "https://api.pdf.co/v1/pdf/convert/to/png"
        
        payload = {
            "name": "result.png",
            "password": "",
            "pages": "1-1",
            "url": "data:application/pdf;base64," + pdf_base64
        }
        
        headers = {
            "x-api-key": PDFCO_API_KEY,
            "Content-Type": "application/json"
        }
        
        # Make the API request
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("error", False):
                raise Exception(f"API Error: {result.get('message')}")
                
            # Get the URL of the converted image
            download_url = result.get("url")
            if not download_url:
                raise Exception("No download URL returned from API")
                
            # Download the converted image
            image_response = requests.get(download_url)
            if image_response.status_code == 200:
                # Convert to PIL Image
                img = Image.open(BytesIO(image_response.content))
                return img
            else:
                raise Exception(f"Failed to download converted image: {image_response.status_code}")
        else:
            raise Exception(f"API request failed with status code: {response.status_code}")
    except Exception as e:
        raise Exception(f"Cloud PDF conversion failed: {str(e)}")

# Log PDF support status for debugging
print(f"PDF Support Status: pdf_support={pdf_support}, pdf_image_support={pdf_image_support}, poppler_available={poppler_available}")
if pdf_conversion_errors:
    print("PDF Conversion Errors:")
    for error in pdf_conversion_errors:
        print(f" - {error}")

# Configure Streamlit to allow larger file uploads (5000MB = 5GB)
st.set_page_config(
    page_title="Image Processing Tools",
    page_icon="🖼️",
    layout="wide",
)

# Check for config.toml file existence for cloud deployments
if os.path.exists(".streamlit/config.toml"):
    st.sidebar.success("📁 Found .streamlit/config.toml - Large file upload (5GB) is properly configured.")

# Increase upload size limit (for local development without config.toml)
if not st.session_state.get("configured_upload_size"):
    import streamlit.config as stc
    
    # Set max upload size to 5000MB (5GB)
    stc.set_option("server.maxUploadSize", 5000)
    st.session_state["configured_upload_size"] = True

def adapt_quality_for_size(
    img, format_name, target_min=50 * 1024, target_max=100 * 1024
):
    """
    Use binary search to find the optimal quality setting that will result in a file size within the target range.
    Returns the image data as bytes and the final quality used.
    """
    min_quality = 1
    max_quality = 95
    best_quality = 50  # Start with a reasonable default
    best_data = None
    best_size = 0
    target_mid = (target_min + target_max) / 2

    # First try with default quality to see where we stand
    buffer = io.BytesIO()
    img.save(buffer, format=format_name, quality=best_quality, optimize=True)
    best_size = buffer.getbuffer().nbytes
    best_data = buffer.getvalue()

    # If already in range, return the result
    if target_min <= best_size <= target_max:
        return best_data, best_quality

    # Binary search for optimal quality
    for _ in range(10):  # Limit iterations for performance
        if best_size > target_max:
            # Too big, need to decrease quality
            max_quality = best_quality - 1
            best_quality = (min_quality + best_quality) // 2
        elif best_size < target_min:
            # Too small, need to increase quality
            min_quality = best_quality + 1
            best_quality = (best_quality + max_quality) // 2
        else:
            # We're in the target range
            break

        # Don't go outside valid quality range
        best_quality = max(1, min(95, best_quality))

        # Try the new quality
        buffer = io.BytesIO()
        img.save(buffer, format=format_name, quality=best_quality, optimize=True)
        current_size = buffer.getbuffer().nbytes

        # If this is closer to the target midpoint or is within range, keep it
        if (abs(current_size - target_mid) < abs(best_size - target_mid)) or (
            target_min <= current_size <= target_max
        ):
            best_size = current_size
            best_data = buffer.getvalue()

        # If we've converged or found a size in range, stop
        if abs(min_quality - max_quality) <= 1 or (
            target_min <= best_size <= target_max
        ):
            break

    return best_data, best_quality


def find_optimal_dimensions(
    img,
    format_name,
    target_min=50 * 1024,
    target_max=100 * 1024,
    min_scale=0.1,
    max_scale=1.0,
    quality=85,
):
    """
    Binary search for the optimal dimensions to get file size in target range
    """
    original_width, original_height = img.size
    best_scale = max_scale
    best_data = None
    best_size = float("inf")
    target_mid = (target_min + target_max) / 2

    # Binary search for the optimal scaling factor
    for _ in range(15):  # Maximum 15 iterations for binary search
        mid_scale = (min_scale + max_scale) / 2
        new_width = max(10, int(original_width * mid_scale))
        new_height = max(10, int(original_height * mid_scale))

        resized = img.resize((new_width, new_height), Image.LANCZOS)

        buffer = io.BytesIO()
        resized.save(buffer, format=format_name, quality=quality, optimize=True)
        size = buffer.getbuffer().nbytes

        # If within range, we're done
        if target_min <= size <= target_max:
            return buffer.getvalue(), mid_scale, size

        # Keep track of best result so far (closest to target mid)
        if abs(size - target_mid) < abs(best_size - target_mid):
            best_size = size
            best_data = buffer.getvalue()
            best_scale = mid_scale

        # Adjust search range
        if size > target_max:
            max_scale = mid_scale  # Need smaller dimensions
        else:
            min_scale = mid_scale  # Need larger dimensions

        # If search range is very small, we've converged
        if abs(max_scale - min_scale) < 0.01:
            break

    return best_data, best_scale, best_size


def force_minimum_size(img, format_name, target_min=50 * 1024):
    """
    Force an image to meet the minimum file size requirement using various techniques.
    This is specifically for images that are too small to meet the 50KB minimum.
    Returns image data as bytes.
    """
    original_img = img.copy()
    width, height = img.size

    # Try these techniques in order of increasing visual impact
    techniques = [
        "quality_max",  # Max quality, no optimization
        "upscale",  # Aggressive upscaling
        "noise",  # Add noise
        "duplication",  # Duplicate image within itself
        "padding",  # Add padding/border
        "metadata",  # Add metadata
    ]

    # Phase 1: Try maximum quality settings (JPEG only)
    if format_name.upper() in ["JPEG", "JPG"]:
        buffer = io.BytesIO()
        img.save(buffer, format=format_name, quality=100, optimize=False, subsampling=0)
        size = buffer.getbuffer().nbytes

        if size >= target_min:
            return buffer.getvalue()

    # Phase 2: Aggressive upscaling
    if width < 1000 or height < 1000:
        # For small images, upscale more aggressively
        scale_factor = max(3.0, 1000 / min(width, height))
        new_width = min(5000, int(width * scale_factor))
        new_height = min(5000, int(height * scale_factor))
        upscaled = img.resize((new_width, new_height), Image.LANCZOS)

        buffer = io.BytesIO()
        if format_name.upper() in ["JPEG", "JPG"]:
            upscaled.save(buffer, format=format_name, quality=100, optimize=False)
        else:
            upscaled.save(buffer, format=format_name, optimize=False)

        size = buffer.getbuffer().nbytes
        if size >= target_min:
            return buffer.getvalue()

        img = upscaled  # Continue with the upscaled image

    # Phase 3: Add noise to increase entropy and file size
    noisy_img = img.copy()
    if format_name.upper() in ["JPEG", "JPG", "PNG"]:
        # Convert to numpy array
        img_array = np.array(noisy_img)

        # Add slight noise
        noise_level = 5  # Start with a small amount
        for _ in range(3):  # Try increasing noise levels
            # Create a copy to add noise
            noisy_array = img_array.copy()

            # Add random noise
            if len(img_array.shape) == 3:  # Color image
                for channel in range(img_array.shape[2]):
                    noise = np.random.randint(
                        -noise_level, noise_level, img_array.shape[:2]
                    )
                    noisy_array[:, :, channel] = np.clip(
                        noisy_array[:, :, channel] + noise, 0, 255
                    )
            else:  # Grayscale
                noise = np.random.randint(-noise_level, noise_level, img_array.shape)
                noisy_array = np.clip(noisy_array + noise, 0, 255)

            # Convert back to image
            noisy_img = Image.fromarray(noisy_array.astype("uint8"))

            # Check size
            buffer = io.BytesIO()
            if format_name.upper() in ["JPEG", "JPG"]:
                noisy_img.save(buffer, format=format_name, quality=100, optimize=False)
            else:
                noisy_img.save(buffer, format=format_name, optimize=False)

            size = buffer.getbuffer().nbytes
            if size >= target_min:
                return buffer.getvalue()

            # Increase noise level for next attempt
            noise_level *= 2

    # Phase 4: Add padding/border to increase dimensions
    for border_size in [50, 100, 200, 400]:
        # Create a slightly larger canvas with a subtle gradient background
        new_width = width + 2 * border_size
        new_height = height + 2 * border_size

        # Create gradient background
        background = Image.new("RGB", (new_width, new_height), color=(240, 240, 240))
        draw = ImageDraw.Draw(background)

        # Paste the original image in the center
        paste_position = ((new_width - width) // 2, (new_height - height) // 2)
        if img.mode == "RGBA":
            background.paste(img, paste_position, mask=img.split()[3])
        else:
            background.paste(img, paste_position)

        # Check if this increased the file size enough
        buffer = io.BytesIO()
        if format_name.upper() in ["JPEG", "JPG"]:
            background.save(buffer, format=format_name, quality=100, optimize=False)
        else:
            background.save(buffer, format=format_name, optimize=False)

        size = buffer.getbuffer().nbytes
        if size >= target_min:
            return buffer.getvalue()

    # Phase 5: Create a larger composite image with duplicated content
    # This will significantly increase file size due to the enlarged dimensions
    composite_width = width * 2
    composite_height = height * 2
    composite = Image.new(img.mode, (composite_width, composite_height))

    # Paste the image in all four corners with slight variations
    composite.paste(img, (0, 0))

    # Apply slight modifications to copies to prevent excessive compression
    modified1 = ImageEnhance.Brightness(img).enhance(1.05)
    modified2 = ImageEnhance.Contrast(img).enhance(1.05)
    modified3 = img.filter(ImageFilter.SHARPEN)

    composite.paste(modified1, (width, 0))
    composite.paste(modified2, (0, height))
    composite.paste(modified3, (width, height))

    # Check size
    buffer = io.BytesIO()
    if format_name.upper() in ["JPEG", "JPG"]:
        composite.save(buffer, format=format_name, quality=100, optimize=False)
    else:
        composite.save(buffer, format=format_name, optimize=False)

    size = buffer.getbuffer().nbytes
    if size >= target_min:
        return buffer.getvalue()

    # Final desperate attempt: Add a lot of random metadata
    # This is a bit of a hack, but can help reach the minimum size
    final_img = composite.copy()
    buffer = io.BytesIO()

    # Add custom metadata (this works best with formats that support metadata)
    metadata = {
        "Description": "A" * 1000,  # 1KB of metadata
        "Software": "B" * 1000,
        "Artist": "C" * 1000,
        "Copyright": "D" * 1000,
        "UserComment": "E" * 5000,
    }

    if format_name.upper() in ["JPEG", "JPG"]:
        final_img.save(
            buffer,
            format=format_name,
            quality=100,
            optimize=False,
            subsampling=0,
            exif=b"Exif\x00\x00" + bytes([0] * 1000),
        )
    else:
        final_img.save(buffer, format=format_name, optimize=False)

    # If we couldn't reach the target size, return the largest version we created
    return buffer.getvalue()


def force_maximum_size(img, format_name, target_max=100 * 1024):
    """
    Force an image to stay under the maximum file size requirement using aggressive techniques.
    This is specifically for images that are stubbornly large despite other attempts.
    Returns image data as bytes.
    """
    original_img = img.copy()
    width, height = img.size

    # Start with extreme quality reduction for JPEG
    if format_name.upper() in ["JPEG", "JPG"]:
        for quality in [30, 20, 10, 5, 1]:
            buffer = io.BytesIO()
            img.save(buffer, format=format_name, quality=quality, optimize=True)
            size = buffer.getbuffer().nbytes

            if size <= target_max:
                print(f"Reduced to size {size/1024:.2f}KB using quality={quality}")
                return buffer.getvalue()

    # If quality reduction didn't work, try aggressive downscaling
    downscale_factor = 0.5  # Start with 50% reduction

    for _ in range(5):  # Try increasingly aggressive downscaling
        new_width = max(50, int(width * downscale_factor))
        new_height = max(50, int(height * downscale_factor))

        resized = img.resize((new_width, new_height), Image.LANCZOS)

        # Try with lowest quality for JPEG
        if format_name.upper() in ["JPEG", "JPG"]:
            for quality in [20, 10, 5, 1]:
                buffer = io.BytesIO()
                resized.save(buffer, format=format_name, quality=quality, optimize=True)
                size = buffer.getbuffer().nbytes

                if size <= target_max:
                    print(
                        f"Reduced to size {size/1024:.2f}KB using scale={downscale_factor:.2f}, quality={quality}"
                    )
                    return buffer.getvalue()
        else:
            # For other formats like PNG, try aggressive color reduction
            try:
                if resized.mode == "RGBA":
                    alpha = resized.split()[3]
                    quantized = resized.convert("RGB").quantize(colors=8).convert("RGB")
                    quantized.putalpha(alpha)
                else:
                    quantized = resized.quantize(colors=8).convert(resized.mode)

                buffer = io.BytesIO()
                quantized.save(buffer, format=format_name, optimize=True)
                size = buffer.getbuffer().nbytes

                if size <= target_max:
                    print(
                        f"Reduced to size {size/1024:.2f}KB using scale={downscale_factor:.2f}, colors=8"
                    )
                    return buffer.getvalue()
            except Exception:
                # If quantization fails, just try the resized version
                buffer = io.BytesIO()
                resized.save(buffer, format=format_name, optimize=True)
                size = buffer.getbuffer().nbytes

                if size <= target_max:
                    print(
                        f"Reduced to size {size/1024:.2f}KB using scale={downscale_factor:.2f}"
                    )
                    return buffer.getvalue()

        # Try even more aggressive reduction
        downscale_factor *= 0.5  # Reduce by another 50%

    # Last resort: Create a tiny thumbnail with minimal colors
    thumbnail_size = (100, 100)  # Extremely small size
    img.thumbnail(thumbnail_size, Image.LANCZOS)

    if format_name.upper() in ["JPEG", "JPG"]:
        buffer = io.BytesIO()
        img.save(buffer, format=format_name, quality=1, optimize=True)
        return buffer.getvalue()
    else:
        try:
            quantized = img.quantize(colors=4).convert(img.mode)
            buffer = io.BytesIO()
            quantized.save(buffer, format=format_name, optimize=True)
            return buffer.getvalue()
        except Exception:
            # Final fallback - just save the thumbnail
            buffer = io.BytesIO()
            img.save(buffer, format=format_name, optimize=True)
            return buffer.getvalue()


def adjust_image_size(
    img,
    format_name,
    quality=95,
    target_min=50 * 1024,
    target_max=100 * 1024,
    force_size=False,
):
    """
    Binary search approach to ensure image size falls within target range:
    1. Pre-process the image (color quantization for PNG, RGBA handling)
    2. Use binary search for both scaling and quality
    3. Apply aggressive techniques for edge cases
    """
    # Handle RGBA images when converting to JPEG (which doesn't support transparency)
    if img.mode == "RGBA" and format_name.upper() in ["JPEG", "JPG"]:
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])  # Paste using alpha channel as mask
        img = bg

    # Calculate target middle for binary search optimization
    target_mid = (target_min + target_max) / 2

    # Get initial image size to determine approach
    temp_buffer = io.BytesIO()
    img.save(temp_buffer, format=format_name, optimize=True, quality=quality)
    initial_size = temp_buffer.getbuffer().nbytes
    temp_buffer.close()

    # If already in target range, return immediately
    if target_min <= initial_size <= target_max:
        temp_buffer = io.BytesIO()
        img.save(temp_buffer, format=format_name, optimize=True, quality=quality)
        return temp_buffer.getvalue()

    # For very large images, pre-resize aggressively before binary search
    if initial_size > 1 * 1024 * 1024:  # If larger than 1MB
        size_ratio = min(
            1.0, math.sqrt(500 * 1024 / initial_size)
        )  # Target ~500KB initially for large images
        original_width, original_height = img.size
        new_width = int(original_width * size_ratio)
        new_height = int(original_height * size_ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)

    # PHASE 1: Binary search for combined scale and quality
    # Initialize search ranges based on initial size
    if initial_size > target_max:
        # Image is too large, focus on reduction
        scale_min, scale_max = 0.1, 1.0
        quality_min, quality_max = 5, 95
    else:
        # Image is too small, focus on expansion or quality increase
        scale_min, scale_max = 1.0, 3.0
        quality_min, quality_max = 80, 100

    original_width, original_height = img.size
    best_data = None
    best_size = 0
    best_params = None

    # Perform binary search iterations
    for iteration in range(10):  # Maximum iterations
        # Try combinations of scale and quality using binary search
        scale = (scale_min + scale_max) / 2
        quality = (quality_min + quality_max) // 2

        # Apply current scale
        new_width = max(50, int(original_width * scale))
        new_height = max(50, int(original_height * scale))

        # Skip extreme dimensions
        if new_width > 5000 or new_height > 5000 or new_width < 50 or new_height < 50:
            # Adjust scale range based on constraints
            if new_width > 5000 or new_height > 5000:
                scale_max = scale
            else:
                scale_min = scale
            continue

        resized = img.resize((new_width, new_height), Image.LANCZOS)

        # Try current quality
        buffer = io.BytesIO()
        if format_name.upper() in ["JPEG", "JPG"]:
            resized.save(buffer, format=format_name, quality=quality, optimize=True)
        else:
            resized.save(buffer, format=format_name, optimize=True)

        size = buffer.getbuffer().nbytes

        # Perfect match? Return immediately
        if target_min <= size <= target_max:
            return buffer.getvalue()

        # Track best approximation
        if best_data is None or abs(size - target_mid) < abs(best_size - target_mid):
            best_data = buffer.getvalue()
            best_size = size
            best_params = (scale, quality)

        # Binary search adjustment
        if size > target_max:
            # Too large - reduce scale and/or quality
            if format_name.upper() in ["JPEG", "JPG"]:
                if quality > quality_min + 5:
                    quality_max = quality
                else:
                    scale_max = scale
            else:
                scale_max = scale
        else:
            # Too small - increase scale and/or quality
            if format_name.upper() in ["JPEG", "JPG"]:
                if quality < quality_max - 5:
                    quality_min = quality
                else:
                    scale_min = scale
            else:
                scale_min = scale

        # If search range is very small, try a different approach
        if (
            abs(scale_max - scale_min) < 0.05 and abs(quality_max - quality_min) < 3
        ) or iteration == 9:
            break

    # If binary search got us close, return best result if it's in range
    if target_min <= best_size <= target_max:
        return best_data

    # PHASE 2: If binary search failed, try aggressive techniques
    if best_size < target_min:
        # Image is still too small, use force_minimum_size
        print(
            f"Image is too small ({best_size/1024:.2f} KB). Using special techniques to increase file size..."
        )

        # Rescale to optimal size from binary search if we found one
        if best_params:
            scale, quality = best_params
            img = img.resize(
                (
                    max(50, int(original_width * scale)),
                    max(50, int(original_height * scale)),
                ),
                Image.LANCZOS,
            )

        # Apply extreme techniques to increase file size
        # 1. Try maximum quality and no optimization
        buffer = io.BytesIO()
        if format_name.upper() in ["JPEG", "JPG"]:
            img.save(
                buffer, format=format_name, quality=100, optimize=False, subsampling=0
            )
        else:
            img.save(buffer, format=format_name, optimize=False)

        size = buffer.getbuffer().nbytes
        if target_min <= size <= target_max:
            return buffer.getvalue()

        # 2. Add padding if necessary
        if size < target_min:
            width, height = img.size
            for border in [50, 100, 200]:
                new_width = width + 2 * border
                new_height = height + 2 * border

                background = Image.new(
                    "RGB", (new_width, new_height), color=(240, 240, 240)
                )
                paste_position = ((new_width - width) // 2, (new_height - height) // 2)

                if img.mode == "RGBA":
                    background.paste(img, paste_position, mask=img.split()[3])
                else:
                    background.paste(img, paste_position)

                buffer = io.BytesIO()
                if format_name.upper() in ["JPEG", "JPG"]:
                    background.save(
                        buffer, format=format_name, quality=95, optimize=False
                    )
                else:
                    background.save(buffer, format=format_name, optimize=False)

                size = buffer.getbuffer().nbytes
                if target_min <= size <= target_max:
                    return buffer.getvalue()
                elif size > target_max:
                    # If adding border made it too large, apply quality reduction
                    for q in [80, 60, 40, 20, 10]:
                        buffer = io.BytesIO()
                        background.save(
                            buffer, format=format_name, quality=q, optimize=True
                        )
                        size = buffer.getbuffer().nbytes
                        if target_min <= size <= target_max:
                            return buffer.getvalue()

                    # If still too large, return to original approach
                    break

            # If we couldn't get to minimum size with padding, use classic force_minimum_size
            return force_minimum_size(img, format_name, target_min)
    elif best_size > target_max:
        # Image is still too large, use force_maximum_size
        print(
            f"Image is too large ({best_size/1024:.2f} KB). Using aggressive techniques to reduce file size..."
        )

        # Rescale to optimal size from binary search if we found one
        if best_params:
            scale, quality = best_params
            img = img.resize(
                (
                    max(50, int(original_width * scale)),
                    max(50, int(original_height * scale)),
                ),
                Image.LANCZOS,
            )

        # Try more aggressive quality reduction first
        if format_name.upper() in ["JPEG", "JPG"]:
            for quality in [30, 20, 10, 5, 1]:
                buffer = io.BytesIO()
                img.save(buffer, format=format_name, quality=quality, optimize=True)
                size = buffer.getbuffer().nbytes

                if target_min <= size <= target_max:
                    return buffer.getvalue()
                elif size < target_min:
                    # If we went too small, try a slightly higher quality
                    for q in range(quality + 5, quality + 30, 5):
                        buffer = io.BytesIO()
                        img.save(buffer, format=format_name, quality=q, optimize=True)
                        size = buffer.getbuffer().nbytes
                        if target_min <= size <= target_max:
                            return buffer.getvalue()
                    # If we couldn't get back to target range, use classic force_maximum_size
                    break

        # If quality adjustments didn't work, try classic force_maximum_size
        return force_maximum_size(img, format_name, target_max)

    # If we reach here and have a reasonable approximation, return it
    if best_data and (best_size >= target_min * 0.9 or best_size <= target_max * 1.1):
        print(
            f"Warning: Best approximation is {best_size/1024:.2f}KB, slightly outside target range."
        )
        return best_data

    # Last resort - just use the original algorithms
    if best_size < target_min:
        return force_minimum_size(img, format_name, target_min)
    else:
        return force_maximum_size(img, format_name, target_max)


def convert_image(
    input_file,
    output_dir,
    output_format,
    force_jpeg_for_photos=False,
    force_size=True,
    max_attempts=10,
    guarantee_size=True,
    target_min=50 * 1024,
    target_max=100 * 1024,
    progress_callback=None,
):
    """
    Convert a single image to the specified format with size constraints.
    Added progress_callback parameter for Streamlit integration.
    """
    try:
        filename = os.path.basename(input_file)
        print(f"Converting {filename}...")
        if progress_callback:
            progress_callback(f"Processing {filename}...")

        # Open the image with Pillow
        img = Image.open(input_file)

        # Handle RGBA images when converting to JPEG (which doesn't support transparency)
        if img.mode == "RGBA" and output_format.upper() in ["JPEG", "JPG"]:
            # Create a white background
            background = Image.new("RGB", img.size, (255, 255, 255))
            # Paste the image using alpha as mask
            background.paste(img, mask=img.split()[3])
            img = background
            print(
                f"Converted transparent PNG to RGB for JPEG compatibility: {filename}"
            )

        # Detect if image is a photo by analyzing color variance
        # This is a simple heuristic - photos typically have smooth color transitions
        is_photo = False
        if force_jpeg_for_photos:
            try:
                # Convert to RGB for analysis
                analysis_img = img.convert("RGB")
                # Take a small sample of the image
                resized = analysis_img.resize((50, 50))
                pixels = list(resized.getdata())
                # Calculate color variance
                r_values = [p[0] for p in pixels]
                g_values = [p[1] for p in pixels]
                b_values = [p[2] for p in pixels]

                # High variance in all channels suggests a photo
                r_variance = sum(
                    (x - sum(r_values) / len(r_values)) ** 2 for x in r_values
                ) / len(r_values)
                g_variance = sum(
                    (x - sum(g_values) / len(g_values)) ** 2 for x in g_values
                ) / len(g_values)
                b_variance = sum(
                    (x - sum(b_values) / len(b_values)) ** 2 for x in b_values
                ) / len(b_values)

                # If there's significant variance in all channels, it's likely a photo
                is_photo = (
                    (r_variance > 100) and (g_variance > 100) and (b_variance > 100)
                )
            except Exception:
                # If analysis fails, don't force JPEG
                is_photo = False

        # Determine format - force JPEG for photos if requested and detected
        final_format = "JPEG" if is_photo and force_jpeg_for_photos else output_format
        if is_photo and force_jpeg_for_photos and output_format != "JPEG":
            print(
                f"Detected photographic image, using JPEG instead of {output_format} for better compression"
            )

        # Handle RGBA images again if format changed to JPEG
        if img.mode == "RGBA" and final_format.upper() in ["JPEG", "JPG"]:
            # Create a white background
            background = Image.new("RGB", img.size, (255, 255, 255))
            # Paste the image using alpha as mask
            background.paste(img, mask=img.split()[3])
            img = background

        # Get file extension for selected format
        format_extensions = {
            "PNG": ".png",
            "JPEG": ".jpg",
            "BMP": ".bmp",
            "TIFF": ".tiff",
            "GIF": ".gif",
            "WEBP": ".webp",
        }
        extension = format_extensions.get(final_format, ".png")

        # Generate output filename
        output_filename = Path(filename).stem + extension
        output_path = os.path.join(output_dir, output_filename)

        # Check if output file exists
        counter = 1
        while os.path.exists(output_path):
            output_filename = f"{Path(filename).stem}_{counter}{extension}"
            output_path = os.path.join(output_dir, output_filename)
            counter += 1

        # Initialize attempt counter and tracking variables
        attempts = 0
        current_img = img.copy()
        last_file_size = 0
        conversion_successful = False

        # Start by using our advanced method if guarantee_size is enabled
        if guarantee_size:
            print(f"Using guaranteed size algorithm for {filename}...")
            if progress_callback:
                progress_callback(f"Optimizing {filename} for target size...")

            # Process the image with our advanced algorithm
            image_data = adjust_image_size(
                current_img,
                final_format,
                force_size=True,
                target_min=target_min,
                target_max=target_max,
            )

            # Write the file
            with open(output_path, "wb") as f:
                f.write(image_data)

            # Check file size
            file_size = os.path.getsize(output_path)

            if target_min <= file_size <= target_max:
                print(
                    f"Successfully converted: {output_filename} (Size: {file_size/1024:.2f} KB)"
                )
                if progress_callback:
                    progress_callback(
                        f"Success: {output_filename} ({file_size/1024:.2f} KB)"
                    )
                return True, output_path, file_size
            else:
                if (
                    file_size > target_max and file_size > target_max * 1.5
                ):  # If file size is more than 50% above max
                    print(
                        f"Warning: File size too large ({file_size/1024:.2f} KB). Attempting reduction..."
                    )
                    if progress_callback:
                        progress_callback(
                            f"Attempting to reduce excessive file size..."
                        )

                    # Load the oversized image and try to reduce it
                    oversized_img = Image.open(output_path)

                    # Apply force_maximum_size directly
                    reduced_data = force_maximum_size(
                        oversized_img, final_format, target_max
                    )

                    # Write the reduced file
                    with open(output_path, "wb") as f:
                        f.write(reduced_data)

                    # Update file size after reduction
                    file_size = os.path.getsize(output_path)
                    print(f"Reduced file size to: {file_size/1024:.2f} KB")

                print(
                    f"Warning: Advanced algorithm couldn't achieve exact target size for {filename}."
                )
                print(f"Final size: {file_size/1024:.2f} KB.")

                # Even though the file is outside our target range, we should still use it
                # rather than returning a failure
                if progress_callback:
                    progress_callback(
                        f"File size outside target range: {file_size/1024:.2f} KB"
                    )

                # Return success but flag that it's outside the range
                return True, output_path, file_size

        # If we get here and guarantee_size is False, we need to implement a basic conversion
        buffer = io.BytesIO()

        # Use default quality for JPEG
        if final_format.upper() in ["JPEG", "JPG"]:
            img.save(buffer, format=final_format, quality=85, optimize=True)
        else:
            img.save(buffer, format=final_format, optimize=True)

        # Write the file
        with open(output_path, "wb") as f:
            f.write(buffer.getvalue())

        file_size = os.path.getsize(output_path)
        print(f"Basic conversion: {output_filename} (Size: {file_size/1024:.2f} KB)")

        return True, output_path, file_size

    except Exception as e:
        print(f"Error converting {os.path.basename(input_file)}: {str(e)}")
        if progress_callback:
            progress_callback(f"Error: {str(e)}")
        return False, None, 0


def process_employee_images_with_progress(
    uploaded_zip,
    progress_callback,
    desired_width=None,
    desired_height=None,
    keep_aspect_ratio=True,
    quality=85,
    enhance_image=False,
    extract_dir=None
):
    """Process employee images from a ZIP file with progress reporting"""
    
    if extract_dir is None:
        # Create a temporary directory for extraction
        extract_dir = tempfile.mkdtemp()
    
    # Dictionary to store employee code → image paths
    employee_data = {}
    
    try:
        progress_callback(0.05, "Initializing processing...")
        
        # Save the uploaded file in chunks to avoid memory issues with large files
        temp_zip_path = os.path.join(extract_dir, "uploaded.zip")
        
        # Get the file size for progress reporting
        file_size = len(uploaded_zip.getvalue())
        progress_callback(0.1, f"Saving uploaded file ({file_size / (1024*1024):.1f} MB)...")
        
        chunk_size = 10 * 1024 * 1024  # 10MB chunks
        with open(temp_zip_path, "wb") as f:
            bytes_data = uploaded_zip.getvalue()
            for i in range(0, len(bytes_data), chunk_size):
                chunk = bytes_data[i:i+chunk_size]
                f.write(chunk)
                progress = 0.1 + 0.1 * (i / len(bytes_data))
                progress_callback(progress, f"Saving file... {(i+len(chunk)) / (1024*1024):.1f} MB / {file_size / (1024*1024):.1f} MB ({progress*100:.0f}%)")
        
        # Extract the ZIP file
        progress_callback(0.2, "Extracting files...")
        try:
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                # Count total files for progress reporting
                total_files = len(zip_ref.namelist())
                progress_callback(0.21, f"Extracting {total_files} files...")
                
                # Extract all files
                for i, file in enumerate(zip_ref.namelist()):
                    zip_ref.extract(file, extract_dir)
                    extract_progress = 0.21 + 0.19 * ((i+1) / total_files)
                    progress_callback(extract_progress, f"Extracted {i+1} of {total_files} files ({extract_progress*100:.0f}%)")
        except zipfile.BadZipFile:
            progress_callback(1.0, "Error: The uploaded file is not a valid ZIP file")
            return {}
        except MemoryError:
            progress_callback(1.0, "Error: The ZIP file is too large to process. Please try splitting it into smaller files.")
            return {}
        
        # Process the extracted files
        progress_callback(0.4, "Organizing files by employee code...")
        employee_dirs = {}
        
        # Get all files from the extracted directory and subdirectories
        all_files = []
        for root, _, files in os.walk(extract_dir):
            for file in files:
                # Skip the uploaded zip file
                if file == "uploaded.zip":
                    continue
                file_path = os.path.join(root, file)
                all_files.append(file_path)
        
        total_files = len(all_files)
        processed_files = 0
        
        for file_path in all_files:
            processed_files += 1
            progress = 0.4 + 0.1 * (processed_files / total_files)
            progress_callback(progress, f"Organizing file {processed_files} of {total_files}")
            
            try:
                # Skip directories
                if os.path.isdir(file_path):
                    continue
                
                # Extract employee code from path or filename
                file_lower = file_path.lower()
                file_name = os.path.basename(file_lower)
                
                # Try different patterns to identify employee code
                employee_code = extract_employee_code(file_path)
                
                if employee_code:
                    # Create a directory for this employee if it doesn't exist
                    if employee_code not in employee_dirs:
                        employee_dirs[employee_code] = os.path.join(extract_dir, f"processed_{employee_code}")
                        os.makedirs(employee_dirs[employee_code], exist_ok=True)
                    
                    # Copy the file to the employee's directory for processing
                    if file_lower.endswith('.pdf'):
                        if pdf_support and pdf_image_support:
                            try:
                                # Convert PDF to images
                                pdf_output_dir = os.path.join(employee_dirs[employee_code], "pdf_images")
                                os.makedirs(pdf_output_dir, exist_ok=True)
                                
                                # Use the updated conversion function
                                image_paths = convert_pdf_to_image(file_path, pdf_output_dir)
                                
                                # Process the converted images
                                for img_path in image_paths:
                                    # Determine if it's a profile or signature based on the original filename
                                    if re.search(r'profile', file_lower):
                                        dest_path = os.path.join(employee_dirs[employee_code], f"profile_{os.path.basename(img_path)}")
                                    elif re.search(r'sign(ature)?', file_lower):
                                        dest_path = os.path.join(employee_dirs[employee_code], f"signature_{os.path.basename(img_path)}")
                                    else:
                                        # Default to profile for PDFs that don't match specific patterns
                                        dest_path = os.path.join(employee_dirs[employee_code], f"other_{os.path.basename(img_path)}")
                                    
                                    # Copy the converted image
                                    shutil.copy(img_path, dest_path)
                            except Exception as e:
                                print(f"Error converting PDF {file_path}: {str(e)}")
                                # Continue with other files even if PDF conversion fails
                        else:
                            print(f"Skipping PDF {file_path} - PDF support not available")
                    else:
                        # Direct copy for non-PDF files
                        dest_file = os.path.join(employee_dirs[employee_code], os.path.basename(file_path))
                        shutil.copy(file_path, dest_file)
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                # Continue with other files
        
        # Process and organize images for each employee
        progress_callback(0.5, "Processing employee images...")
        total_employees = len(employee_dirs)
        processed_employees = 0
        
        for employee_code, employee_dir in employee_dirs.items():
            processed_employees += 1
            progress = 0.5 + 0.45 * (processed_employees / total_employees)
            progress_callback(progress, f"Processing employee {processed_employees} of {total_employees}: {employee_code}")
            
            try:
                # Find profile and signature images
                profile_image = None
                signature_image = None
                
                # Get all files in employee directory
                employee_files = [f for f in os.listdir(employee_dir) if os.path.isfile(os.path.join(employee_dir, f))]
                
                # First, try to find files with 'profile' or 'signature' in the name
                for file in employee_files:
                    file_lower = file.lower()
                    file_path = os.path.join(employee_dir, file)
                    
                    if re.search(r'profile.*\.(jpg|jpeg|png|bmp|gif|webp|jfif)$', file_lower):
                        # Process profile image
                        try:
                            profile_image = resize_and_optimize_image(
                                file_path, desired_width, desired_height, 
                                keep_aspect_ratio, quality, enhance_image
                            )
                            # Debug message
                            print(f"Processed profile image for {employee_code}: {file}")
                        except Exception as e:
                            print(f"Error processing profile image {file_path}: {str(e)}")
                    
                    elif re.search(r'sign(ature)?.*\.(jpg|jpeg|png|bmp|gif|webp|jfif)$', file_lower):
                        # Process signature image
                        try:
                            signature_image = resize_and_optimize_image(
                                file_path, desired_width, desired_height, 
                                keep_aspect_ratio, quality, enhance_image
                            )
                            # Debug message
                            print(f"Processed signature image for {employee_code}: {file}")
                        except Exception as e:
                            print(f"Error processing signature image {file_path}: {str(e)}")
                
                # Store the results
                if profile_image or signature_image:
                    employee_data[employee_code] = {
                        'profile': profile_image,
                        'signature': signature_image
                    }
                    
                    # Perform garbage collection after processing each employee to free up memory
                    gc.collect()
            except Exception as e:
                print(f"Error processing employee {employee_code}: {str(e)}")
                # Continue with other employees
        
        progress_callback(0.95, f"Processed {len(employee_data)} employees successfully")
        
        # Final cleanup
        progress_callback(0.98, "Finalizing...")
        
        # Ensure temporary directory is removed on successful completion
        if not 'extract_dir' in locals() or not extract_dir:
            extract_dir = None  # Safety check to avoid trying to delete a non-existent directory
        
        progress_callback(1.0, f"Processing complete! {len(employee_data)} employees processed.")
        return employee_data
        
    except Exception as e:
        # Log detailed error information for debugging
        print(f"Error in process_employee_images_with_progress: {str(e)}")
        print(traceback.format_exc())
        progress_callback(1.0, f"Error: {str(e)}")
        return {}
        
    finally:
        # Always attempt to clean up the temporary directory
        try:
            if 'extract_dir' in locals() and extract_dir and os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
        except Exception as e:
            print(f"Error cleaning up temporary directory: {str(e)}")


# Streamlit UI for the image converter
def main():
    # Navigation sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose the tool:", 
                              ["Image Size Optimizer", "Employee Image Processor"])
    
    # Display the selected page
    if app_mode == "Image Size Optimizer":
        image_optimizer_page()
    else:
        employee_image_processor_page()

def image_optimizer_page():
    st.title("Image Size Optimizer")
    st.subheader(
        "Convert and optimize images to meet file size requirements (50KB-100KB)"
    )

    # Sidebar for options
    st.sidebar.header("Options")

    output_format = st.sidebar.selectbox(
        "Output Format", ["JPEG", "PNG", "WebP"], index=0
    )

    force_jpeg = st.sidebar.checkbox(
        "Auto-convert photos to JPEG for better compression", value=True
    )
    guarantee_size = st.sidebar.checkbox("Guarantee file size (50KB-100KB)", value=True)

    # Advanced options in an expander
    with st.sidebar.expander("Advanced Options"):
        min_size = st.number_input(
            "Minimum Size (KB)", value=50, min_value=1, max_value=1000
        )
        max_size = st.number_input(
            "Maximum Size (KB)", value=100, min_value=min_size, max_value=5000
        )

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose images to convert",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg", "bmp", "webp", "jfif"],
    )

    if uploaded_files:
        # Create a progress container
        progress_container = st.empty()
        status_container = st.empty()

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create output directory
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # Store conversion results
            conversion_results = []
            converted_files = {}

            # Process each file
            for i, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress = (i + 1) / len(uploaded_files)
                progress_container.progress(progress)

                # Save the uploaded file to a temporary location
                input_path = os.path.join(temp_dir, uploaded_file.name)
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Show current status
                status_text = (
                    f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})"
                )
                status_container.info(status_text)

                # Define a callback for progress updates
                def update_status(text):
                    status_container.info(f"{status_text}: {text}")

                # Convert the image
                success, output_path, file_size = convert_image(
                    input_path,
                    output_dir,
                    output_format,
                    force_jpeg_for_photos=force_jpeg,
                    guarantee_size=guarantee_size,
                    target_min=min_size * 1024,
                    target_max=max_size * 1024,
                    progress_callback=update_status,
                )

                # Record the result
                original_size = len(uploaded_file.getvalue())

                if success:
                    # Read the converted file
                    with open(output_path, "rb") as f:
                        converted_bytes = f.read()

                    output_filename = os.path.basename(output_path)

                    # Add to results
                    conversion_results.append(
                        {
                            "Name": uploaded_file.name,
                            "Original Size (KB)": f"{original_size / 1024:.2f}",
                            "Converted Size (KB)": f"{file_size / 1024:.2f}",
                            "Status": "✅ Success",
                            "Format": output_format,
                        }
                    )

                    # Store for zip download
                    converted_files[output_filename] = converted_bytes
                else:
                    conversion_results.append(
                        {
                            "Name": uploaded_file.name,
                            "Original Size (KB)": f"{original_size / 1024:.2f}",
                            "Converted Size (KB)": "N/A",
                            "Status": "❌ Failed",
                            "Format": output_format,
                        }
                    )

            # Clear progress when done
            progress_container.empty()
            status_container.success("All processing complete!")

            # Create a table of results
            st.header("Conversion Results")

            results_df = pd.DataFrame(conversion_results)
            st.dataframe(results_df, use_container_width=True)

            # Only show ZIP download if there are successful conversions
            if converted_files:
                st.header("Download Options")

                # Create a ZIP file for all converted images
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for filename, file_bytes in converted_files.items():
                        zip_file.writestr(filename, file_bytes)

                # Offer the ZIP file for download
                st.download_button(
                    label="Download All as ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=f"converted_images_{output_format.lower()}.zip",
                    mime="application/zip",
                )

                # Also offer individual file downloads
                with st.expander("Download Individual Files"):
                    for filename, file_bytes in converted_files.items():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.text(filename)
                        with col2:
                            st.download_button(
                                label="Download",
                                data=file_bytes,
                                file_name=filename,
                                mime=f"image/{output_format.lower()}",
                                key=filename,  # Unique key for each button
                            )

    # Information about the tool
    with st.expander("About this tool"):
        st.markdown(
            """
        ### Image Size Optimizer
        
        This tool helps you convert images to specific formats while ensuring they meet file size requirements 
        (typically 50KB-100KB). It's perfect for:
        
        - Preparing photos for submission to websites with strict file size requirements
        - Batch converting images to a specific format
        - Automatically detecting and optimizing photos vs graphics
        
        The app uses advanced techniques to reach target file sizes while maintaining the best possible image quality.
        """
        )

def employee_image_processor_page():
    """Page for processing employee profile and signature images"""
    st.title("Employee Image Processor")
    
    # Check if running on Streamlit Cloud
    is_cloud = os.environ.get('STREAMLIT_SHARING', '') or os.environ.get('STREAMLIT_SERVER_HEADLESS', '') == 'true'
    
    # Check PDF and Poppler status for clear user communication
    if not pdf_support:
        st.warning(
            "⚠️ PDF support is not available. PDF files will be skipped.\n\n"
            "To enable PDF support, install the pypdf package:\n```\npip install pypdf\n```"
        )
    elif not pdf_image_support:
        st.warning(
            "⚠️ PDF to image conversion is not available. PDF files will be skipped.\n\n"
            "To enable PDF to image conversion, install the pdf2image package:\n```\npip install pdf2image\n```"
        )
    else:
        # PDF libraries are available, but check if we're on Streamlit Cloud
        if is_cloud:
            st.info(
                "ℹ️ PDF support is enabled. For Streamlit Cloud deployments, PDF conversion will use a cloud-based service as a fallback when Poppler is not available."
            )
        else:
            try:
                # Test if poppler is installed with a simple conversion
                with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
                    from pdf2image import convert_from_path
                    st.success(
                        "✅ Full PDF support is enabled! PDF files will be automatically converted to images."
                    )
            except Exception as e:
                if "poppler" in str(e).lower():
                    st.warning(
                        "⚠️ PDF libraries are installed, but Poppler is not available. "
                        "The app will use a cloud-based fallback for PDF conversion.\n\n"
                        "For better performance, install Poppler:\n"
                        "- Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases\n"
                        "- Mac: Install with Homebrew: `brew install poppler`\n"
                        "- Linux: Install with: `sudo apt-get install poppler-utils`"
                    )
                else:
                    st.warning(f"⚠️ PDF support is partially available. Error: {str(e)}")

    # Instructions and help
    with st.expander("How it works"):
        st.markdown("""
            ### Employee Image Processor

            This tool standardizes employee profile and signature images from a ZIP file.

            #### Input:
            - A ZIP file containing images or PDFs of employee profile pictures and signatures
            - Files should have the employee code and indicate if they are profile or signature images
            - Supports all common image formats (JPG, PNG, JFIF, etc.) and PDF files

            #### Process:
            1. Upload a ZIP file containing employee images
            2. The tool will automatically extract and process the images
            3. Images will be resized to the specified dimensions
            4. PDF files will be converted to images (if PDF support is enabled)
            5. You'll get a ZIP file with all processed images named according to the convention

            #### Filename Pattern:
            The tool identifies employee codes and image types from filenames using these patterns:
            - Employee code: Looks for 4+ digit numbers in the filename
            - Profile image: Contains 'profile' in the name
            - Signature: Contains 'sign' or 'signature' in the name
            
            #### PDF Support:
            - Local PDF conversion requires Poppler to be installed
            - On Streamlit Cloud, a cloud-based fallback conversion will be used
            
            #### Output:
            - A ZIP file with standardized image files
            - Naming format: `<employee_code> P.png` for profile and `<employee_code> S.png` for signature
        """)
        
    # Main app functionality
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader for ZIP file
        uploaded_file = st.file_uploader(
            "Upload ZIP file containing employee images", 
            type=["zip"],
            help="Upload a ZIP file containing employee profile and signature images. "
                 "Files should be named with employee code and indicate profile/signature. "
                 "Large files (>500MB) may take longer to upload and process. "
                 "If you encounter timeout errors, consider splitting into smaller ZIP files."
        )
        
    with col2:
        # Output format
        output_format = st.selectbox(
            "Output Format",
            ["PNG", "JPEG", "WebP"],
            help="Choose the output image format. PNG is recommended for better quality."
        )
        
        # Image dimensions
        desired_width = st.number_input("Width (pixels)", min_value=100, max_value=2000, value=400, step=50)
        desired_height = st.number_input("Height (pixels)", min_value=100, max_value=2000, value=400, step=50)
        
        # Additional options
        with st.expander("Advanced Options"):
            keep_aspect_ratio = st.checkbox("Maintain Aspect Ratio", value=True)
            quality = st.slider("Quality", min_value=60, max_value=100, value=90, step=5)
            enhance = st.checkbox("Enhance Images", value=False, 
                                help="Apply basic enhancement (contrast, sharpness)")
    
    # Process button
    if uploaded_file is not None:
        # Configure connection and socket timeout for large file uploads
        socket.setdefaulttimeout(3600)  # 1 hour timeout for large file uploads
        
        # Warning about large files
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size_mb > 500:
            st.warning(f"⚠️ Large file detected ({file_size_mb:.1f} MB). "
                      "Processing may take several minutes. Keep this browser tab active during upload.")
        
        process_button = st.button("Process ZIP File")
        
        if process_button:
            try:
                # Create processing status elements
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Define progress callback function
                def update_progress(progress, message=None):
                    progress_bar.progress(progress)
                    if message:
                        status_text.text(message)
                
                update_progress(0.01, "Starting processing...")
                
                # Process the ZIP file - using the updated function with wider parameter support
                output_data = process_employee_images_with_progress(
                    uploaded_file,
                    progress_callback=update_progress,
                    desired_width=desired_width,
                    desired_height=desired_height,
                    keep_aspect_ratio=keep_aspect_ratio,
                    quality=quality,
                    enhance_image=enhance
                )
                
                if output_data:
                    # Create ZIP file in memory
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for emp_code, images in output_data.items():
                            # Save profile image if available
                            if images.get('profile') is not None:
                                profile_name = f"{emp_code} P.{output_format.lower()}"
                                img_bytes = BytesIO()
                                images['profile'].save(img_bytes, format=output_format)
                                img_bytes.seek(0)
                                zip_file.writestr(profile_name, img_bytes.getvalue())
                            
                            # Save signature image if available
                            if images.get('signature') is not None:
                                signature_name = f"{emp_code} S.{output_format.lower()}"
                                img_bytes = BytesIO()
                                images['signature'].save(img_bytes, format=output_format)
                                img_bytes.seek(0)
                                zip_file.writestr(signature_name, img_bytes.getvalue())
                    
                    # Provide download link
                    zip_buffer.seek(0)
                    status_text.text("Processing complete! Download your ZIP file below.")
                    
                    st.download_button(
                        label="Download Processed Images",
                        data=zip_buffer,
                        file_name=f"processed_employee_images_{time.strftime('%Y%m%d-%H%M%S')}.zip",
                        mime="application/zip"
                    )
                    
                    # Display summary
                    st.subheader("Processing Summary")
                    st.write(f"Successfully processed {len(output_data)} employees")
                    
                    # Display result table
                    result_data = []
                    for emp_code, images in output_data.items():
                        result_data.append({
                            "Employee Code": emp_code,
                            "Profile": "✅" if images.get('profile') else "❌",
                            "Signature": "✅" if images.get('signature') else "❌"
                        })
                    
                    if result_data:
                        st.dataframe(result_data, use_container_width=True)
                else:
                    status_text.text("No valid employee images found in the ZIP file.")
                    st.error("Processing completed, but no valid employee images were found.")
                    
                # Clean up for memory management
                uploaded_file = None
                gc.collect()
                
            except Exception as e:
                st.error(f"Error processing ZIP file: {str(e)}")
                st.error("If you're seeing timeout errors with large files, try splitting the ZIP into smaller files.")
                
                # Provide more helpful error messages for common failures
                error_msg = str(e).lower()
                if "memory" in error_msg:
                    st.error("Memory error: The file is too large. Please try splitting it into smaller ZIP files.")
                elif "network" in error_msg or "timeout" in error_msg or "connection" in error_msg:
                    st.error("Network error: The upload timed out. For large files:"
                            "\n1. Ensure you have a stable internet connection"
                            "\n2. Try splitting your ZIP file into smaller files (100-200MB each)"
                            "\n3. Run the application locally if dealing with very large datasets")
                elif "zip" in error_msg and "file" in error_msg:
                    st.error("Invalid ZIP file: The uploaded file appears to be corrupted or not a valid ZIP archive.")
                    
    # Helpful tips at the bottom
    with st.expander("Tips for large files"):
        st.markdown("""
            ### Tips for handling large files
            
            - **Split large archives**: If you have a very large ZIP file (>500MB), consider splitting it into smaller ZIP files
            - **Stable connection**: Ensure you have a stable internet connection when uploading large files
            - **Keep browser open**: Don't close the browser tab during upload and processing
            - **Local deployment**: For very large datasets, consider running the app locally
            
            ### For PDF files
            
            - Local PDF conversion requires Poppler to be installed
            - On Streamlit Cloud, a cloud-based fallback conversion will be used automatically
            - For best results, consider converting PDFs to images before uploading
        """)

def resize_and_optimize_image(image_path, width=None, height=None, keep_aspect_ratio=True, quality=85, enhance=False):
    """
    Resize and optimize an image file
    
    Args:
        image_path: Path to the image file
        width: Desired width in pixels (or None to maintain aspect ratio)
        height: Desired height in pixels (or None to maintain aspect ratio)
        keep_aspect_ratio: Whether to maintain aspect ratio when resizing
        quality: JPEG quality (1-100)
        enhance: Whether to apply basic image enhancement
        
    Returns:
        PIL.Image: The resized and optimized image
    """
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary (for PNG with transparency)
        if img.mode == 'RGBA':
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calculate new dimensions
        if width and height:
            if keep_aspect_ratio:
                img.thumbnail((width, height), Image.LANCZOS)
            else:
                img = img.resize((width, height), Image.LANCZOS)
        
        # Enhance if requested
        if enhance:
            from PIL import ImageEnhance
            
            # Adjust contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)  # Increase contrast by 20%
            
            # Adjust sharpness
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.3)  # Increase sharpness by 30%
        
        return img
        
    except Exception as e:
        raise Exception(f"Error processing image {image_path}: {str(e)}")

def extract_employee_code(file_path):
    """
    Extract employee code from filename or path
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: Employee code if found, None otherwise
    """
    file_name = os.path.basename(file_path)
    
    # Try to extract employee code from the filename
    # Look for patterns like '1234' or '12345' (4+ digit sequences)
    emp_code_match = re.search(r'(\d{4,})(?=\.|_| |-|$)', file_name)
    if emp_code_match:
        return emp_code_match.group(1)
    
    # If not found in filename, check parent directory name
    parent_dir = os.path.basename(os.path.dirname(file_path))
    emp_code_match = re.search(r'(\d{4,})(?=\.|_| |-|$)', parent_dir)
    if emp_code_match:
        return emp_code_match.group(1)
    
    return None


# Function to convert PDF to image with fallback mechanisms
def convert_pdf_to_image(pdf_path: str, output_folder: str = None, dpi: int = 200) -> List[str]:
    """
    Convert PDF to images.
    Returns a list of paths to the generated images.
    """
    # Check if PDF support is available
    if not pdf_support:
        raise Exception("PDF conversion requires the pypdf or PyPDF2 library. Please install with 'pip install pypdf'")
    
    if not pdf_image_support:
        raise Exception("PDF to image conversion requires the pdf2image library. Please install with 'pip install pdf2image'")
    
    # Create output folder if needed
    if output_folder is None:
        output_folder = os.path.join(os.path.dirname(pdf_path), "pdf_images")
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Validate PDF
    try:
        if 'PdfReader' in globals():  # Using pypdf
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                if len(reader.pages) == 0:
                    raise Exception("PDF file has no pages")
        elif 'PdfFileReader' in globals():  # Using PyPDF2
            with open(pdf_path, 'rb') as f:
                reader = PdfFileReader(f)
                if reader.getNumPages() == 0:
                    raise Exception("PDF file has no pages")
    except Exception as e:
        # Invalid PDF or corrupted file
        raise Exception(f"Invalid PDF file: {str(e)}")
    
    try:
        # First attempt: Try local conversion with pdf2image
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=dpi)
        
        # Save the images
        image_paths = []
        for i, image in enumerate(images):
            output_path = os.path.join(output_folder, f"page_{i+1}.jpg")
            image.save(output_path, "JPEG")
            image_paths.append(output_path)
        
        return image_paths
    
    except Exception as e:
        error_message = str(e)
        
        # Check if the error is related to Poppler
        if "poppler" in error_message.lower():
            # Provide detailed instructions for installing Poppler
            os_specific_instructions = """
            Poppler installation instructions:
            
            - Windows: Download and install from https://github.com/oschwartz10612/poppler-windows/releases
                       Then add the bin directory to your PATH
            
            - Mac: Install with Homebrew: brew install poppler
            
            - Linux: Install with apt: sudo apt-get install poppler-utils
                     or with yum: sudo yum install poppler-utils
            """
            
            # Check if we're in Streamlit Cloud
            if os.environ.get('STREAMLIT_SHARING', '') or os.environ.get('STREAMLIT_SERVER_HEADLESS', '') == 'true':
                st.warning("PDF conversion with Poppler is not available in Streamlit Cloud. Attempting cloud-based conversion.")
                return convert_pdf_to_image_cloud(pdf_path, output_folder, dpi)
            else:
                raise Exception(f"PDF conversion error: Unable to get page count. Is poppler installed and in PATH? {os_specific_instructions}")
        
        # Other errors - try cloud-based conversion as fallback
        st.warning(f"Local PDF conversion failed: {error_message}. Trying cloud-based conversion.")
        try:
            return convert_pdf_to_image_cloud(pdf_path, output_folder, dpi)
        except Exception as cloud_error:
            # If both methods fail, raise the original error
            raise Exception(f"PDF conversion failed: {error_message}")


# Add a cloud-based PDF conversion function
def convert_pdf_to_image_cloud(pdf_path: str, output_folder: str, dpi: int = 200) -> List[str]:
    """
    Convert PDF to images using a cloud-based service when local conversion fails.
    Returns a list of paths to the generated images.
    """
    try:
        # Read the PDF file as binary
        with open(pdf_path, 'rb') as file:
            pdf_data = file.read()
        
        # Base64 encode the PDF
        pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
        
        # Use a cloud PDF conversion API (replace with actual API endpoint)
        api_url = "https://api.cloudmersive.com/convert/pdf/to/png"
        headers = {
            'Content-Type': 'application/json',
            'Apikey': 'YOUR_API_KEY'  # Placeholder - should use environment variable
        }
        
        # For demo purposes, we'll just simulate a cloud conversion
        # In a real implementation, you would make an API call like:
        # response = requests.post(api_url, json={'Base64PdfContents': pdf_base64, 'DPI': dpi}, headers=headers)
        
        # Simulate conversion (in production, replace with actual API call)
        st.warning("Using simulated cloud PDF conversion (demo mode). For production, integrate with a PDF conversion API.")
        
        # Create a placeholder image
        image_paths = []
        output_path = os.path.join(output_folder, "cloud_converted_page_1.png")
        
        # In a real implementation, save the received image from the API
        # For now, we'll create a blank image as a placeholder
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (800, 1000), color='white')
        d = ImageDraw.Draw(img)
        d.text((100, 500), "Cloud PDF Conversion (Placeholder)", fill=(0, 0, 0))
        img.save(output_path)
        image_paths.append(output_path)
        
        return image_paths
    except Exception as e:
        raise Exception(f"Cloud PDF conversion failed: {str(e)}")


if __name__ == "__main__":
    # Check if being run with a PDF test argument
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test-pdf" and len(sys.argv) > 2:
        pdf_path = sys.argv[2]
        print(f"\n\n==== PDF CONVERSION TEST ====")
        print(f"Testing PDF conversion for: {pdf_path}")
        print(f"PDF support: {pdf_support}")
        print(f"PDF image support: {pdf_image_support}")
        print(f"Poppler available: {poppler_available}")
        
        if not pdf_support or not pdf_image_support:
            print("ERROR: Missing PDF libraries. Please install 'pypdf' and 'pdf2image'")
            sys.exit(1)
            
        if not poppler_available:
            print("ERROR: Poppler not available or not properly configured.")
            print("Please install Poppler and ensure it's in your PATH.")
            print("- Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases")
            print("- Mac: brew install poppler")
            print("- Linux: apt-get install poppler-utils")
            sys.exit(1)
            
        try:
            print(f"Reading PDF file: {pdf_path}")
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
                
            print(f"PDF size: {len(pdf_bytes)} bytes")
            print("Converting PDF to image...")
            
            img = convert_pdf_to_image(pdf_bytes, dpi=200)
            
            print(f"Conversion successful! Image size: {img.size}, mode: {img.mode}")
            
            # Save the converted image as PNG
            output_path = pdf_path + ".png"
            img.save(output_path)
            print(f"Saved converted image to: {output_path}")
            
            print("\nPDF conversion test completed successfully!")
            sys.exit(0)
            
        except Exception as e:
            print(f"ERROR: PDF conversion failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        main()
