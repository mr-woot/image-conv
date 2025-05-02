import os
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import io
from pathlib import Path
import math
import tempfile
import numpy as np
from io import BytesIO
import streamlit as st
import zipfile
import re
import shutil
import pandas as pd

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


def process_employee_images_with_progress(uploaded_zip, output_format="PNG", progress_callback=None):
    """
    Process a zip file containing employee image archives with progress updates
    
    Args:
        uploaded_zip: The uploaded parent zip file
        output_format: The output format for the images (default: PNG)
        progress_callback: Callback function for progress updates
        
    Returns:
        BytesIO: ZIP buffer containing processed images with standard naming
        list: Results of processing for display
    """
    results = []
    
    if progress_callback:
        progress_callback("Initializing processing...", 0.05)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the uploaded zip file
        parent_zip_path = os.path.join(temp_dir, "uploaded.zip")
        
        # Process in chunks to avoid memory issues with very large files
        if progress_callback:
            progress_callback("Saving uploaded file to temporary storage...", 0.08)
            
        try:
            # For very large files, write in chunks
            chunk_size = 10 * 1024 * 1024  # 10MB chunks
            with open(parent_zip_path, "wb") as f:
                # Get file size for progress reporting
                total_size = len(uploaded_zip.getvalue())
                bytes_written = 0
                
                # Reset file pointer
                uploaded_zip.seek(0)
                
                # Write in chunks
                while True:
                    chunk = uploaded_zip.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_written += len(chunk)
                    
                    # Update progress for large files
                    if total_size > 100 * 1024 * 1024 and progress_callback:  # 100MB
                        save_progress = 0.05 + (0.05 * (bytes_written / total_size))
                        progress_callback(f"Saving file: {bytes_written/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB", save_progress)
            
            # Free up memory by clearing the uploaded file from memory after saving it
            uploaded_data = None
            if hasattr(uploaded_zip, '_close'):
                # Only used for memory management, not closing the file
                uploaded_data = uploaded_zip.getvalue()
                uploaded_zip.seek(0)  # Reset position
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error saving file: {str(e)}", 0.08)
            raise Exception(f"Failed to save uploaded file: {str(e)}")
            
        if progress_callback:
            progress_callback("Extracting main ZIP file...", 0.1)
            
        # Create extraction and output directories
        extract_dir = os.path.join(temp_dir, "extracted")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(extract_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Process in chunks for large files
        try:
            # Extract the parent zip file
            with zipfile.ZipFile(parent_zip_path, 'r') as parent_zip:
                file_count = len(parent_zip.namelist())
                
                if progress_callback:
                    progress_callback(f"Extracting {file_count} files...", 0.15)
                
                # For very large files, extract in chunks
                if file_count > 1000:
                    for i, file in enumerate(parent_zip.namelist()):
                        try:
                            parent_zip.extract(file, extract_dir)
                            # Periodically report progress
                            if i % 100 == 0 and progress_callback:
                                extract_progress = 0.15 + (0.2 * (i / file_count))
                                progress_callback(f"Extracted {i}/{file_count} files...", extract_progress)
                                
                                # Force garbage collection periodically for very large extractions
                                if i % 1000 == 0:
                                    import gc
                                    gc.collect()
                        except Exception as e:
                            # Log error but continue with other files
                            if progress_callback:
                                progress_callback(f"Warning: Failed to extract {file}: {str(e)}", None)
                else:
                    parent_zip.extractall(extract_dir)
        except zipfile.BadZipFile:
            if progress_callback:
                progress_callback("Error: The file is not a valid ZIP file or is corrupted.", 0.1)
            raise zipfile.BadZipFile("The uploaded file is not a valid ZIP file or is corrupted")
        except MemoryError:
            if progress_callback:
                progress_callback("Error: Not enough memory to extract this ZIP file. Please try a smaller file.", 0.1)
            raise MemoryError("Not enough memory to extract the ZIP file. Try splitting it into smaller files.")
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error extracting ZIP: {str(e)}", 0.1)
            raise
        
        if progress_callback:
            progress_callback("Looking for ZIP files and images...", 0.4)
        
        # Look for zip files in the extracted directory first
        employee_zips = []
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith('.zip'):
                    employee_zips.append(os.path.join(root, file))
        
        # If we found employee ZIP files, process them
        if employee_zips:
            if progress_callback:
                progress_callback(f"Found {len(employee_zips)} employee ZIP files to process", 0.45)
            
            # Process each employee zip file
            for i, emp_zip_path in enumerate(employee_zips):
                try:
                    # Update progress
                    if progress_callback:
                        emp_progress = 0.45 + (0.45 * (i / max(1, len(employee_zips))))
                        progress_callback(f"Processing employee ZIP {i+1}/{len(employee_zips)}", emp_progress)
                    
                    # Extract employee code from zip filename
                    emp_code = os.path.splitext(os.path.basename(emp_zip_path))[0]
                    
                    # Create a temporary directory for this employee's files
                    emp_dir = os.path.join(temp_dir, emp_code)
                    os.makedirs(emp_dir, exist_ok=True)
                    
                    # Extract the employee zip file
                    with zipfile.ZipFile(emp_zip_path, 'r') as emp_zip:
                        emp_zip.extractall(emp_dir)
                    
                    # Find profile and signature images/pdfs
                    profile_img_path = None
                    signature_img_path = None
                    profile_is_pdf = False
                    signature_is_pdf = False
                    
                    for root, _, files in os.walk(emp_dir):
                        for file in files:
                            file_lower = file.lower()
                            # Check for image files first
                            if re.search(r'profile.*\.(jpg|jpeg|png|bmp|gif|webp|jfif)$', file_lower):
                                profile_img_path = os.path.join(root, file)
                            elif re.search(r'sign(ature)?.*\.(jpg|jpeg|png|bmp|gif|webp|jfif)$', file_lower):
                                signature_img_path = os.path.join(root, file)
                    
                    # If image not found, look for PDF files
                    if not profile_img_path:
                        for root, _, files in os.walk(emp_dir):
                            for file in files:
                                file_lower = file.lower()
                                if re.search(r'profile.*\.pdf$', file_lower):
                                    profile_img_path = os.path.join(root, file)
                                    profile_is_pdf = True
                                    break
                    
                    if not signature_img_path:
                        for root, _, files in os.walk(emp_dir):
                            for file in files:
                                file_lower = file.lower()
                                if re.search(r'sign(ature)?.*\.pdf$', file_lower):
                                    signature_img_path = os.path.join(root, file)
                                    signature_is_pdf = True
                                    break
                            
                    # Process profile image if found
                    if profile_img_path:
                        try:
                            profile_output_name = f"{emp_code} P.{output_format.lower()}"
                            
                            # Handle PDF conversion if needed
                            if profile_is_pdf:
                                if not pdf_support or not pdf_image_support:
                                    results.append({
                                        "Employee Code": emp_code,
                                        "Image Type": "Profile (PDF)",
                                        "Status": "❌ PDF Support not available. Install 'pypdf' and 'pdf2image' packages.",
                                    })
                                    img = None
                                else:
                                    with open(profile_img_path, 'rb') as pdf_file:
                                        pdf_bytes = pdf_file.read()
                                    try:
                                        img = convert_pdf_to_image(pdf_bytes)
                                    except Exception as e:
                                        results.append({
                                            "Employee Code": emp_code,
                                            "Image Type": "Profile (PDF)",
                                            "Status": f"❌ PDF Conversion Failed: {str(e)}",
                                        })
                                        img = None
                            else:
                                img = Image.open(profile_img_path)
                            
                            if img:
                                # Convert to RGB if necessary
                                if img.mode == 'RGBA' and output_format.upper() in ['JPEG', 'JPG']:
                                    bg = Image.new('RGB', img.size, (255, 255, 255))
                                    bg.paste(img, mask=img.split()[3])
                                    img = bg
                                
                                # Save with the new format
                                output_path = os.path.join(output_dir, profile_output_name)
                                img.save(output_path, format=output_format.upper())
                                
                                results.append({
                                    "Employee Code": emp_code,
                                    "Image Type": "Profile" + (" (from PDF)" if profile_is_pdf else ""),
                                    "Status": "✅ Success",
                                })
                                
                                # Free memory immediately
                                img = None
                                import gc
                                gc.collect()
                        except Exception as e:
                            results.append({
                                "Employee Code": emp_code,
                                "Image Type": "Profile" + (" (from PDF)" if profile_is_pdf else ""),
                                "Status": f"❌ Failed: {str(e)}",
                            })
                    else:
                        results.append({
                            "Employee Code": emp_code,
                            "Image Type": "Profile",
                            "Status": "❌ Not found",
                        })
                    
                    # Process signature image if found
                    if signature_img_path:
                        try:
                            signature_output_name = f"{emp_code} S.{output_format.lower()}"
                            
                            # Handle PDF conversion if needed
                            if signature_is_pdf:
                                if not pdf_support or not pdf_image_support:
                                    results.append({
                                        "Employee Code": emp_code,
                                        "Image Type": "Signature (PDF)",
                                        "Status": "❌ PDF Support not available. Install 'pypdf' and 'pdf2image' packages.",
                                    })
                                    img = None
                                else:
                                    with open(signature_img_path, 'rb') as pdf_file:
                                        pdf_bytes = pdf_file.read()
                                    try:
                                        img = convert_pdf_to_image(pdf_bytes)
                                    except Exception as e:
                                        results.append({
                                            "Employee Code": emp_code,
                                            "Image Type": "Signature (PDF)",
                                            "Status": f"❌ PDF Conversion Failed: {str(e)}",
                                        })
                                        img = None
                            else:
                                img = Image.open(signature_img_path)
                            
                            if img:
                                # Convert to RGB if necessary
                                if img.mode == 'RGBA' and output_format.upper() in ['JPEG', 'JPG']:
                                    bg = Image.new('RGB', img.size, (255, 255, 255))
                                    bg.paste(img, mask=img.split()[3])
                                    img = bg
                                
                                # Save with the new format
                                output_path = os.path.join(output_dir, signature_output_name)
                                img.save(output_path, format=output_format.upper())
                                
                                results.append({
                                    "Employee Code": emp_code,
                                    "Image Type": "Signature" + (" (from PDF)" if signature_is_pdf else ""),
                                    "Status": "✅ Success",
                                })
                                
                                # Free memory immediately
                                img = None
                                import gc
                                gc.collect()
                        except Exception as e:
                            results.append({
                                "Employee Code": emp_code,
                                "Image Type": "Signature" + (" (from PDF)" if signature_is_pdf else ""),
                                "Status": f"❌ Failed: {str(e)}",
                            })
                    else:
                        results.append({
                            "Employee Code": emp_code,
                            "Image Type": "Signature",
                            "Status": "❌ Not found",
                        })
                    
                    # Clean up employee directory to save space for large batches
                    try:
                        shutil.rmtree(emp_dir)
                    except:
                        pass
                        
                except Exception as e:
                    results.append({
                        "Employee Code": os.path.splitext(os.path.basename(emp_zip_path))[0],
                        "Image Type": "Processing",
                        "Status": f"❌ Failed: {str(e)}",
                    })
                    
                # Force garbage collection after each employee to prevent memory buildup
                if i % 10 == 0:
                    import gc
                    gc.collect()
        else:
            # If no nested ZIP files found, process images in the extracted directory
            # This handles the case where images are directly in the parent ZIP
            if progress_callback:
                progress_callback("No employee ZIP files found. Processing images directly...", 0.45)
            
            # Group files by employee code
            employee_images = {}
            
            # First pass: find all image files and group by employee code from filename
            image_count = 0
            
            # Process image files
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.jfif')):
                        file_path = os.path.join(root, file)
                        file_lower = file.lower()
                        
                        # Try to extract employee code from the filename
                        emp_code_match = re.search(r'(\d{4,})(?=\.|_)', file_lower)
                        if emp_code_match:
                            emp_code = emp_code_match.group(1)
                            
                            # Initialize entry for this employee if it doesn't exist
                            if emp_code not in employee_images:
                                employee_images[emp_code] = {'profile': None, 'signature': None, 
                                                            'profile_is_pdf': False, 'signature_is_pdf': False}
                            
                            # Determine if it's a profile or signature image
                            if re.search(r'profile', file_lower):
                                employee_images[emp_code]['profile'] = file_path
                            elif re.search(r'sign(ature)?', file_lower):
                                employee_images[emp_code]['signature'] = file_path
                            
                            image_count += 1
                            if progress_callback and image_count % 10 == 0:
                                progress_callback(f"Found {image_count} images...", 0.5)
                                
                                # Periodically free memory
                                if image_count % 100 == 0:
                                    import gc
                                    gc.collect()
            
            # Now search for PDF files where images weren't found
            pdf_count = 0
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        file_path = os.path.join(root, file)
                        file_lower = file.lower()
                        
                        # Try to extract employee code from the filename
                        emp_code_match = re.search(r'(\d{4,})(?=\.|_)', file_lower)
                        if emp_code_match:
                            emp_code = emp_code_match.group(1)
                            
                            # Initialize entry for this employee if it doesn't exist
                            if emp_code not in employee_images:
                                employee_images[emp_code] = {'profile': None, 'signature': None, 
                                                            'profile_is_pdf': False, 'signature_is_pdf': False}
                            
                            # Determine if it's a profile or signature pdf
                            if re.search(r'profile', file_lower) and not employee_images[emp_code]['profile']:
                                employee_images[emp_code]['profile'] = file_path
                                employee_images[emp_code]['profile_is_pdf'] = True
                                pdf_count += 1
                            elif re.search(r'sign(ature)?', file_lower) and not employee_images[emp_code]['signature']:
                                employee_images[emp_code]['signature'] = file_path
                                employee_images[emp_code]['signature_is_pdf'] = True
                                pdf_count += 1
                                
                            if progress_callback and pdf_count % 10 == 0:
                                progress_callback(f"Found {pdf_count} PDF files...", 0.55)
            
            # Process grouped images
            if progress_callback:
                progress_callback(f"Processing {len(employee_images)} employees' images...", 0.6)
                
            for i, (emp_code, images) in enumerate(employee_images.items()):
                if progress_callback:
                    emp_progress = 0.6 + (0.3 * (i / max(1, len(employee_images))))
                    progress_callback(f"Processing employee {i+1}/{len(employee_images)}", emp_progress)
                
                # Process profile image if found
                if images['profile']:
                    try:
                        profile_output_name = f"{emp_code} P.{output_format.lower()}"
                        
                        # Handle PDF conversion if needed
                        if images['profile_is_pdf']:
                            if not pdf_support or not pdf_image_support:
                                results.append({
                                    "Employee Code": emp_code,
                                    "Image Type": "Profile (PDF)",
                                    "Status": "❌ PDF Support not available. Install 'pypdf' and 'pdf2image' packages.",
                                })
                                img = None
                            else:
                                with open(images['profile'], 'rb') as pdf_file:
                                    pdf_bytes = pdf_file.read()
                                try:
                                    img = convert_pdf_to_image(pdf_bytes)
                                except Exception as e:
                                    results.append({
                                        "Employee Code": emp_code,
                                        "Image Type": "Profile (PDF)",
                                        "Status": f"❌ PDF Conversion Failed: {str(e)}",
                                    })
                                    img = None
                        else:
                            img = Image.open(images['profile'])
                        
                        if img:
                            # Convert to RGB if necessary
                            if img.mode == 'RGBA' and output_format.upper() in ['JPEG', 'JPG']:
                                bg = Image.new('RGB', img.size, (255, 255, 255))
                                bg.paste(img, mask=img.split()[3])
                                img = bg
                            
                            # Save with the new format
                            output_path = os.path.join(output_dir, profile_output_name)
                            img.save(output_path, format=output_format.upper())
                            
                            results.append({
                                "Employee Code": emp_code,
                                "Image Type": "Profile" + (" (from PDF)" if images['profile_is_pdf'] else ""),
                                "Status": "✅ Success",
                            })
                            
                            # Free memory immediately
                            img = None
                            import gc
                            gc.collect()
                    except Exception as e:
                        results.append({
                            "Employee Code": emp_code,
                            "Image Type": "Profile" + (" (from PDF)" if images['profile_is_pdf'] else ""),
                            "Status": f"❌ Failed: {str(e)}",
                        })
                else:
                    results.append({
                        "Employee Code": emp_code,
                        "Image Type": "Profile",
                        "Status": "❌ Not found",
                    })
                
                # Process signature image if found
                if images['signature']:
                    try:
                        signature_output_name = f"{emp_code} S.{output_format.lower()}"
                        
                        # Handle PDF conversion if needed
                        if images['signature_is_pdf']:
                            if not pdf_support or not pdf_image_support:
                                results.append({
                                    "Employee Code": emp_code,
                                    "Image Type": "Signature (PDF)",
                                    "Status": "❌ PDF Support not available. Install 'pypdf' and 'pdf2image' packages.",
                                })
                                img = None
                            else:
                                with open(images['signature'], 'rb') as pdf_file:
                                    pdf_bytes = pdf_file.read()
                                try:
                                    img = convert_pdf_to_image(pdf_bytes)
                                except Exception as e:
                                    results.append({
                                        "Employee Code": emp_code,
                                        "Image Type": "Signature (PDF)",
                                        "Status": f"❌ PDF Conversion Failed: {str(e)}",
                                    })
                                    img = None
                        else:
                            img = Image.open(images['signature'])
                        
                        if img:
                            # Convert to RGB if necessary
                            if img.mode == 'RGBA' and output_format.upper() in ['JPEG', 'JPG']:
                                bg = Image.new('RGB', img.size, (255, 255, 255))
                                bg.paste(img, mask=img.split()[3])
                                img = bg
                            
                            # Save with the new format
                            output_path = os.path.join(output_dir, signature_output_name)
                            img.save(output_path, format=output_format.upper())
                            
                            results.append({
                                "Employee Code": emp_code,
                                "Image Type": "Signature" + (" (from PDF)" if images['signature_is_pdf'] else ""),
                                "Status": "✅ Success",
                            })
                            
                            # Free memory immediately
                            img = None
                            import gc
                            gc.collect()
                    except Exception as e:
                        results.append({
                            "Employee Code": emp_code,
                            "Image Type": "Signature" + (" (from PDF)" if images['signature_is_pdf'] else ""),
                            "Status": f"❌ Failed: {str(e)}",
                        })
                else:
                    results.append({
                        "Employee Code": emp_code,
                        "Image Type": "Signature",
                        "Status": "❌ Not found",
                    })
                
                # Force garbage collection after each employee to prevent memory buildup
                if i % 10 == 0:
                    import gc
                    gc.collect()
        
        if progress_callback:
            progress_callback("Creating output ZIP file...", 0.9)
            
        # Create a zip file with all processed images
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as output_zip:
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    # Read in chunks for very large files
                    with open(file_path, 'rb') as f:
                        output_zip.writestr(file, f.read())
        
        if progress_callback:
            progress_callback("Processing complete!", 0.95)
            
        # Force garbage collection before returning
        import gc
        gc.collect()
    
    return zip_buffer, results


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
    st.title("Employee Image Processor")
    st.subheader("Extract and standardize profile and signature images from employee archives")
    
    # Sidebar options
    st.sidebar.header("Options")
    output_format = st.sidebar.selectbox(
        "Output Format", ["PNG", "JPEG", "WebP"], index=0
    )
    
    # Show PDF support status with detailed diagnostics
    if not pdf_support or not pdf_image_support:
        st.error("❌ PDF support is not available. The tool will only process image files.")
        
        # Show specific installation instructions based on what's missing
        with st.expander("How to enable PDF support"):
            st.markdown("""
            ### PDF Support Installation Instructions
            
            To enable PDF processing, you need to install these Python packages:
            
            ```bash
            pip install pypdf pdf2image
            ```
            
            After installation, restart the application.
            """)
            
            if pdf_conversion_errors:
                st.markdown("### Specific errors detected:")
                for error in pdf_conversion_errors:
                    st.markdown(f"- {error}")
    elif not poppler_available:
        st.warning("""
        ⚠️ **PDF support is partially available, but Poppler is missing.** 
        
        PDF files will not be processed correctly until you install Poppler:
        """)
        
        with st.expander("Poppler Installation Instructions"):
            st.markdown("""
            ### Poppler Installation Instructions
            
            Poppler is required for PDF to image conversion. Install it based on your operating system:
            
            - **Windows**: 
              1. Download from [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases)
              2. Extract the ZIP file
              3. Add the `bin` directory to your system PATH
            
            - **Mac**: 
              ```bash
              brew install poppler
              ```
            
            - **Linux (Ubuntu/Debian)**: 
              ```bash
              apt-get install poppler-utils
              ```
            
            After installing Poppler, restart the application.
            """)
            
            if pdf_conversion_errors:
                st.markdown("### Specific errors detected:")
                for error in pdf_conversion_errors:
                    st.markdown(f"- {error}")
    else:
        st.success("✅ PDF support is fully enabled. The application will automatically convert PDF files to images.")
    
    # Information about the tool
    st.markdown("""
    ### How it works:
    
    1. Upload a **ZIP file** containing employee image files
    2. The tool supports two organization methods:
       - **Nested ZIP files**: Each employee's files are in a separate ZIP file named with their employee code (e.g., `489713.zip`)
       - **Direct image files**: Images are directly in the parent ZIP with employee codes in their filenames (e.g., `profile_image10177.jpg`)
    3. For nested ZIP files, the employee code is taken from the ZIP filename
    4. For direct images, the employee code is extracted from the image filename
    5. The tool extracts profile images and signature images based on filename patterns
    6. Supports various image formats including JPG, PNG, JFIF, and others; PDF files with "profile" or "signature" in the name will be automatically converted to images
    7. Files are renamed to a standardized format: `{Employee-Code} P.{format}` and `{Employee-Code} S.{format}`
    8. All processed images are packaged into a single ZIP file for download
    """)
    
    # File upload warning for large files
    st.warning("""
    ### Large File Upload Instructions
    
    For files larger than 500MB:
    1. Ensure you have a stable internet connection
    2. Keep the browser tab active during upload
    3. The upload may take several minutes - be patient and don't close the tab
    4. If you get a timeout or network error, try these solutions:
       - Break up your ZIP file into smaller ZIPs (200-300MB each)
       - Try a different browser (Firefox or Chrome often work best)
       - Run the application locally for better performance
    
    You can upload ZIP files up to 5GB in size (with proper configuration).
    """)
    
    # File uploader with explicit timeout messages
    uploaded_zip = st.file_uploader(
        "Upload ZIP file containing employee archives", 
        type=["zip"],
        help="If you encounter a timeout error with large files, try splitting your file into smaller chunks"
    )
    
    if uploaded_zip:
        file_size_mb = len(uploaded_zip.getvalue()) / (1024 * 1024)
        st.write(f"Uploaded file size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 200:
            st.warning(f"Large file detected ({file_size_mb:.2f} MB). Processing may take several minutes.")
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Processing employee images... This may take a while depending on the size and number of archives."):
            try:
                # Process status updates through a callback
                def progress_callback(message, progress_value=None):
                    status_text.info(message)
                    if progress_value is not None:
                        progress_bar.progress(progress_value)
                
                # Process the uploaded ZIP file with progress updates
                zip_buffer, results = process_employee_images_with_progress(
                    uploaded_zip, 
                    output_format,
                    progress_callback
                )
                
                # Display results
                progress_bar.progress(100)
                status_text.success("Processing complete!")
                
                # Create a table of results
                st.header("Processing Results")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Offer the ZIP file for download
                st.header("Download Processed Images")
                st.download_button(
                    label="Download All Processed Images as ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=f"employee_images_{output_format.lower()}.zip",
                    mime="application/zip",
                )
                
                # Display summary
                total_employees = len(set([r["Employee Code"] for r in results]))
                successful_profiles = len([r for r in results if r["Image Type"] == "Profile" and "Success" in r["Status"]])
                successful_signatures = len([r for r in results if r["Image Type"] == "Signature" and "Success" in r["Status"]])
                
                st.subheader("Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Employees", total_employees)
                with col2:
                    st.metric("Profile Images", f"{successful_profiles}/{total_employees}")
                with col3:
                    st.metric("Signature Images", f"{successful_signatures}/{total_employees}")
                
            except zipfile.BadZipFile:
                st.error("Error: The uploaded file is not a valid ZIP file or is corrupted.")
                st.info("Please check your ZIP file and try again. If the file is very large, it might have been truncated during upload.")
            except MemoryError:
                st.error("Memory Error: The file is too large to process with available memory.")
                st.info("Try splitting your ZIP file into smaller files and process them separately.")
            except Exception as e:
                st.error(f"Error processing the ZIP file: {str(e)}")
                st.info("If you're encountering network timeouts, try these solutions:")
                st.markdown("""
                1. **Split your file** into smaller ZIP files (200-300MB each)
                2. **Try a different browser** (Firefox or Chrome often work best for large uploads)
                3. **Run the application locally** by installing it on your computer
                4. **Check your internet connection** stability
                """)
    else:
        # Show alternative upload methods if no file is uploaded
        with st.expander("Having trouble with large files?"):
            st.markdown("""
            ### Alternatives for very large files:
            
            1. **Split your ZIP** into multiple smaller files (we recommend 200-300MB per ZIP)
            2. **Run the application locally** on your computer:
               ```
               pip install -r requirements.txt
               streamlit run main.py
               ```
            3. **Use a more stable network** connection when uploading
            4. **Contact support** if you continue experiencing issues
            """)
            
    # Help section
    with st.expander("Need Help?"):
        st.markdown("""
        ### Expected Structure Options:
        
        #### Option 1: Nested ZIP files - Employee code from ZIP filename
        ```
        Main.zip
        ├── 10177.zip    # Employee code = 10177
        │   ├── aadhar_card_back.jpg
        │   ├── aadhar_card_front.jpg
        │   ├── profile_image.jpg    # Will be used for Profile image
        │   ├── signature.jpg        # Will be used for Signature image
        │   ├── profile.pdf          # Alternatively, PDFs are supported
        │   ├── signature.pdf        # Alternatively, PDFs are supported
        │   └── ...
        ├── 10178.zip    # Employee code = 10178
        │   ├── profile_image.jpg    # Will be used for Profile image
        │   ├── signature.jpg        # Will be used for Signature image
        │   └── ...
        └── ...
        ```
        
        #### Option 2: Direct image files - Employee code from image filename
        ```
        Main.zip
        ├── profile_image10177.jpg    # Employee code = 10177, Profile image
        ├── signature10177.jpg        # Employee code = 10177, Signature image
        ├── profile10177.pdf          # Employee code = 10177, Profile (PDF)
        ├── sign10178.pdf             # Employee code = 10178, Signature (PDF)
        ├── profile_image10178.jpg    # Employee code = 10178, Profile image
        ├── signature10178.jpg        # Employee code = 10178, Signature image
        └── ...
        ```
        
        ### Important Notes:
        
        1. For nested ZIP files, the employee code is taken directly from the ZIP filename
        2. For direct images, the employee code is extracted from the image filename
        3. The tool supports both image files (.jpg, .jpeg, .png, .jfif, etc.) and PDF files
        4. The tool looks for any files containing "profile" or "sign"/"signature" in their names
        5. PDF files will be automatically converted to images (first page only)
        6. All extracted images are standardized to: `{Employee-Code} P.{format}` and `{Employee-Code} S.{format}`
        7. File size limit is 5GB for the main ZIP file
        """)


# Function to convert PDF to image
def convert_pdf_to_image(pdf_bytes, dpi=200):
    """
    Convert a PDF file to a PIL Image
    
    Args:
        pdf_bytes: PDF file as bytes
        dpi: DPI for conversion (higher = better quality but larger size)
        
    Returns:
        PIL Image of the first page
    """
    if not pdf_support or not pdf_image_support:
        raise ImportError("PDF conversion requires both pypdf/PyPDF2 and pdf2image packages")
        
    try:
        # Check if it's a valid PDF
        try:
            is_valid = False
            if 'PdfReader' in globals():
                reader = PdfReader(BytesIO(pdf_bytes))
                if len(reader.pages) > 0:
                    is_valid = True
            else:
                reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
                if len(reader.pages) > 0:
                    is_valid = True
                    
            if not is_valid:
                raise ValueError("PDF appears to be empty or invalid")
        except Exception as e:
            raise ValueError(f"Not a valid PDF file: {str(e)}")
            
        # Convert the PDF to images
        try:
            # Print debug info
            pdf_size = len(pdf_bytes)
            first_bytes = pdf_bytes[:20]
            print(f"PDF size: {pdf_size} bytes, starts with: {first_bytes}")
            
            images = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=1)
            
            if not images or len(images) == 0:
                raise ValueError("PDF conversion resulted in 0 images")
                
            print(f"Successfully converted PDF to {len(images)} images")
            
        except Exception as e:
            error_message = str(e).lower()
            
            if "poppler" in error_message:
                raise Exception(
                    "Poppler is not installed or not in PATH. "
                    "This dependency is required for PDF conversion. "
                    "Please install Poppler:\n"
                    "- Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases\n"
                    "- Mac: brew install poppler\n"
                    "- Linux (Ubuntu/Debian): apt-get install poppler-utils\n"
                )
            else:
                raise
        
        # Return the first page as an image
        if images and len(images) > 0:
            return images[0]
        else:
            raise ValueError("Could not convert PDF to image")
    except Exception as e:
        raise Exception(f"PDF conversion error: {str(e)}")


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
