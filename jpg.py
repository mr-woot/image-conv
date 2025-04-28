import sys
import os
import argparse
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageDraw
import io
from pathlib import Path
import glob
import math
import tempfile
import random
import numpy as np
from io import BytesIO

def adapt_quality_for_size(img, format_name, target_min=50*1024, target_max=100*1024):
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
        if ((abs(current_size - target_mid) < abs(best_size - target_mid)) or 
            (target_min <= current_size <= target_max)):
            best_size = current_size
            best_data = buffer.getvalue()
            
        # If we've converged or found a size in range, stop
        if abs(min_quality - max_quality) <= 1 or (target_min <= best_size <= target_max):
            break

    return best_data, best_quality

def find_optimal_dimensions(img, format_name, target_min=50*1024, target_max=100*1024, 
                           min_scale=0.1, max_scale=1.0, quality=85):
    """
    Binary search for the optimal dimensions to get file size in target range
    """
    original_width, original_height = img.size
    best_scale = max_scale
    best_data = None
    best_size = float('inf')
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

def force_minimum_size(img, format_name, target_min=50*1024):
    """
    Force an image to meet the minimum file size requirement using various techniques.
    This is specifically for images that are too small to meet the 50KB minimum.
    Returns image data as bytes.
    """
    original_img = img.copy()
    width, height = img.size
    
    # Try these techniques in order of increasing visual impact
    techniques = [
        "quality_max",      # Max quality, no optimization
        "upscale",          # Aggressive upscaling
        "noise",            # Add noise
        "duplication",      # Duplicate image within itself
        "padding",          # Add padding/border
        "metadata",         # Add metadata
    ]
    
    # Phase 1: Try maximum quality settings (JPEG only)
    if format_name.upper() in ['JPEG', 'JPG']:
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
        if format_name.upper() in ['JPEG', 'JPG']:
            upscaled.save(buffer, format=format_name, quality=100, optimize=False)
        else:
            upscaled.save(buffer, format=format_name, optimize=False)
        
        size = buffer.getbuffer().nbytes
        if size >= target_min:
            return buffer.getvalue()
        
        img = upscaled  # Continue with the upscaled image
    
    # Phase 3: Add noise to increase entropy and file size
    noisy_img = img.copy()
    if format_name.upper() in ['JPEG', 'JPG', 'PNG']:
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
                    noise = np.random.randint(-noise_level, noise_level, img_array.shape[:2])
                    noisy_array[:, :, channel] = np.clip(noisy_array[:, :, channel] + noise, 0, 255)
            else:  # Grayscale
                noise = np.random.randint(-noise_level, noise_level, img_array.shape)
                noisy_array = np.clip(noisy_array + noise, 0, 255)
            
            # Convert back to image
            noisy_img = Image.fromarray(noisy_array.astype('uint8'))
            
            # Check size
            buffer = io.BytesIO()
            if format_name.upper() in ['JPEG', 'JPG']:
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
        background = Image.new('RGB', (new_width, new_height), color=(240, 240, 240))
        draw = ImageDraw.Draw(background)
        
        # Paste the original image in the center
        paste_position = ((new_width - width) // 2, (new_height - height) // 2)
        if img.mode == 'RGBA':
            background.paste(img, paste_position, mask=img.split()[3])
        else:
            background.paste(img, paste_position)
        
        # Check if this increased the file size enough
        buffer = io.BytesIO()
        if format_name.upper() in ['JPEG', 'JPG']:
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
    if format_name.upper() in ['JPEG', 'JPG']:
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
        'Description': 'A' * 1000,  # 1KB of metadata
        'Software': 'B' * 1000,
        'Artist': 'C' * 1000,
        'Copyright': 'D' * 1000,
        'UserComment': 'E' * 5000,
    }
    
    if format_name.upper() in ['JPEG', 'JPG']:
        final_img.save(buffer, format=format_name, quality=100, optimize=False, 
                      subsampling=0, exif=b"Exif\x00\x00" + bytes([0] * 1000))
    else:
        final_img.save(buffer, format=format_name, optimize=False)
    
    # If we couldn't reach the target size, return the largest version we created
    return buffer.getvalue()

def force_maximum_size(img, format_name, target_max=100*1024):
    """
    Force an image to stay under the maximum file size requirement using aggressive techniques.
    This is specifically for images that are stubbornly large despite other attempts.
    Returns image data as bytes.
    """
    original_img = img.copy()
    width, height = img.size
    
    # Start with extreme quality reduction for JPEG
    if format_name.upper() in ['JPEG', 'JPG']:
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
        if format_name.upper() in ['JPEG', 'JPG']:
            for quality in [20, 10, 5, 1]:
                buffer = io.BytesIO()
                resized.save(buffer, format=format_name, quality=quality, optimize=True)
                size = buffer.getbuffer().nbytes
                
                if size <= target_max:
                    print(f"Reduced to size {size/1024:.2f}KB using scale={downscale_factor:.2f}, quality={quality}")
                    return buffer.getvalue()
        else:
            # For other formats like PNG, try aggressive color reduction
            try:
                if resized.mode == 'RGBA':
                    alpha = resized.split()[3]
                    quantized = resized.convert('RGB').quantize(colors=8).convert('RGB')
                    quantized.putalpha(alpha)
                else:
                    quantized = resized.quantize(colors=8).convert(resized.mode)
                
                buffer = io.BytesIO()
                quantized.save(buffer, format=format_name, optimize=True)
                size = buffer.getbuffer().nbytes
                
                if size <= target_max:
                    print(f"Reduced to size {size/1024:.2f}KB using scale={downscale_factor:.2f}, colors=8")
                    return buffer.getvalue()
            except Exception:
                # If quantization fails, just try the resized version
                buffer = io.BytesIO()
                resized.save(buffer, format=format_name, optimize=True)
                size = buffer.getbuffer().nbytes
                
                if size <= target_max:
                    print(f"Reduced to size {size/1024:.2f}KB using scale={downscale_factor:.2f}")
                    return buffer.getvalue()
        
        # Try even more aggressive reduction
        downscale_factor *= 0.5  # Reduce by another 50%
    
    # Last resort: Create a tiny thumbnail with minimal colors
    thumbnail_size = (100, 100)  # Extremely small size
    img.thumbnail(thumbnail_size, Image.LANCZOS)
    
    if format_name.upper() in ['JPEG', 'JPG']:
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

def adjust_image_size(img, format_name, quality=95, target_min=50*1024, target_max=100*1024, force_size=False):
    """
    Multi-stage approach to ensure image size falls within target range:
    1. Pre-process the image (color quantization for PNG, RGBA handling)
    2. Try dimension adjustment with binary search
    3. Try quality adjustment with binary search
    4. Combine dimension and quality adjustments if needed
    """
    # Handle RGBA images when converting to JPEG (which doesn't support transparency)
    if img.mode == 'RGBA' and format_name.upper() in ['JPEG', 'JPG']:
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])  # Paste using alpha channel as mask
        img = bg

    # Get initial image size to determine approach
    initial_size = 0
    temp_buffer = io.BytesIO()
    img.save(temp_buffer, format=format_name, optimize=True, quality=quality)
    initial_size = temp_buffer.getbuffer().nbytes
    temp_buffer.close()
    
    # If already in target range, return immediately
    if target_min <= initial_size <= target_max:
        temp_buffer = io.BytesIO()
        img.save(temp_buffer, format=format_name, optimize=True, quality=quality)
        return temp_buffer.getvalue()
    
    # For very large images, pre-resize aggressively
    if initial_size > 1*1024*1024:  # If larger than 1MB
        # Calculate resize ratio based on source size - larger files need more reduction
        size_ratio = min(1.0, math.sqrt(500*1024 / initial_size))  # Target ~500KB initially for large images
        
        # Apply the initial aggressive resize
        original_width, original_height = img.size
        new_width = int(original_width * size_ratio)
        new_height = int(original_height * size_ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # For PNG, try color quantization
    original_img = img.copy()
    if format_name.upper() == 'PNG' and img.mode in ['RGB', 'RGBA']:
        try:
            if img.mode == 'RGBA':
                # Preserve alpha channel
                alpha = img.split()[3]
                img_rgb = img.convert('RGB').quantize(colors=256).convert('RGB')
                img_rgba = img_rgb.copy()
                img_rgba.putalpha(alpha)
                img = img_rgba
            else:
                img = img.quantize(colors=256).convert('RGB')
                
            # Check if quantization helped get into range
            temp_buffer = io.BytesIO()
            img.save(temp_buffer, format=format_name, optimize=True)
            quantized_size = temp_buffer.getbuffer().nbytes
            
            if not (target_min <= quantized_size <= target_max):
                # Try more aggressive quantization if still out of range
                if img.mode == 'RGBA':
                    alpha = original_img.split()[3]
                    img_rgb = original_img.convert('RGB').quantize(colors=128).convert('RGB')
                    img_rgba = img_rgb.copy()
                    img_rgba.putalpha(alpha)
                    img = img_rgba
                else:
                    img = original_img.quantize(colors=128).convert('RGB')
            
            temp_buffer = io.BytesIO()
            img.save(temp_buffer, format=format_name, optimize=True)
            quantized_size = temp_buffer.getbuffer().nbytes
            
            # If quantization made it worse, revert
            if quantized_size > initial_size and not (target_min <= quantized_size <= target_max):
                img = original_img
        except Exception:
            # If quantization fails, revert to original
            img = original_img
    
    # PHASE 1: Try adjusting dimensions first with a reasonable quality
    if force_size:
        optimal_data, scale_used, size_achieved = find_optimal_dimensions(
            img, format_name, target_min, target_max, 
            min_scale=0.1, max_scale=1.0, quality=85
        )
        
        if target_min <= size_achieved <= target_max:
            return optimal_data
    
    # PHASE 2: Try adjusting quality with original dimensions
    if format_name.upper() in ['JPEG', 'JPG']:
        optimal_data, quality_used = adapt_quality_for_size(img, format_name, target_min, target_max)
        
        # Check if we found a solution
        temp_buffer = io.BytesIO()
        temp_buffer.write(optimal_data)
        size = temp_buffer.getbuffer().nbytes
        
        if target_min <= size <= target_max:
            return optimal_data
    
    # PHASE 3: If still not in range, try combined approach: resize + quality adjustment
    # Start by trying different scales with binary search
    min_scale = 0.1
    max_scale = 1.0
    original_width, original_height = img.size
    
    for attempt in range(5):  # Limit number of attempts
        # Try a new scale in the middle of our current range
        scale = (min_scale + max_scale) / 2
        new_width = max(10, int(original_width * scale))
        new_height = max(10, int(original_height * scale))
        
        # Skip if dimensions are too small
        if new_width < 10 or new_height < 10:
            break
            
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # For JPG, try adjusting quality at this scale
        if format_name.upper() in ['JPEG', 'JPG']:
            optimal_data, quality_used = adapt_quality_for_size(resized_img, format_name, target_min, target_max)
            
            # Check size
            temp_buffer = io.BytesIO()
            temp_buffer.write(optimal_data)
            size = temp_buffer.getbuffer().nbytes
            
            if target_min <= size <= target_max:
                return optimal_data
        else:
            # For other formats, just try the current scale
            buffer = io.BytesIO()
            resized_img.save(buffer, format=format_name, optimize=True)
            size = buffer.getbuffer().nbytes
            
            if target_min <= size <= target_max:
                return buffer.getvalue()
                
        # Adjust the scale range based on current size
        if size > target_max:
            max_scale = scale  # Need smaller size
        else:
            min_scale = scale  # Need larger size
            
    # PHASE 4: Last resort - try super aggressive dimension reduction + quality tuning
    if format_name.upper() in ['JPEG', 'JPG']:
        # Try several quality levels with smallest reasonable dimensions
        small_img = img.resize((max(int(original_width * 0.2), 50), max(int(original_height * 0.2), 50)), Image.LANCZOS)
        
        for quality in [80, 60, 40, 30, 20]:
            buffer = io.BytesIO()
            small_img.save(buffer, format=format_name, quality=quality, optimize=True)
            size = buffer.getbuffer().nbytes
            
            if target_min <= size <= target_max:
                return buffer.getvalue()
    elif format_name.upper() == 'PNG':
        # For PNG, try drastic color reduction
        try:
            small_img = img.resize((max(int(original_width * 0.2), 50), max(int(original_height * 0.2), 50)), Image.LANCZOS)
            if img.mode == 'RGBA':
                alpha = small_img.split()[3]
                small_rgb = small_img.convert('RGB').quantize(colors=32).convert('RGB')
                small_rgba = small_rgb.copy()
                small_rgba.putalpha(alpha)
                small_img = small_rgba
            else:
                small_img = small_img.quantize(colors=32).convert('RGB')
            
            buffer = io.BytesIO()
            small_img.save(buffer, format=format_name, optimize=True)
            size = buffer.getbuffer().nbytes
            
            if target_min <= size <= target_max:
                return buffer.getvalue()
        except Exception:
            pass
    
    # Check if current size is too small or too large for requirements
    buffer = io.BytesIO()
    img.save(buffer, format=format_name, optimize=True)
    size = buffer.getbuffer().nbytes
    
    # If we're still below minimum size, use special handling for small images
    if size < target_min:
        print(f"Image is too small ({size/1024:.2f} KB). Using special techniques to increase file size...")
        return force_minimum_size(img, format_name, target_min)
    
    # If we're above maximum size, use special handling for large images
    if size > target_max:
        print(f"Image is too large ({size/1024:.2f} KB). Using aggressive techniques to reduce file size...")
        return force_maximum_size(img, format_name, target_max)
    
    # If all else fails, just return the original image with optimizations
    print(f"Warning: Could not adjust image to size range (50KB-100KB). Final size: {size/1024:.2f} KB")
    buffer = io.BytesIO()
    img.save(buffer, format=format_name, optimize=True)
    return buffer.getvalue()
    
def convert_image(input_file, output_dir, output_format, force_jpeg_for_photos=False, force_size=False, skip_size_check=False, max_attempts=5, guarantee_size=False):
    """Convert a single image to the specified format with size constraints."""
    try:
        filename = os.path.basename(input_file)
        print(f"Converting {filename}...")
        
        # Open the image with Pillow
        img = Image.open(input_file)
        
        # Handle RGBA images when converting to JPEG (which doesn't support transparency)
        if img.mode == 'RGBA' and output_format.upper() in ['JPEG', 'JPG']:
            # Create a white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            # Paste the image using alpha as mask
            background.paste(img, mask=img.split()[3])
            img = background
            print(f"Converted transparent PNG to RGB for JPEG compatibility: {filename}")
        
        # Detect if image is a photo by analyzing color variance
        # This is a simple heuristic - photos typically have smooth color transitions
        is_photo = False
        if force_jpeg_for_photos:
            try:
                # Convert to RGB for analysis
                analysis_img = img.convert('RGB')
                # Take a small sample of the image
                resized = analysis_img.resize((50, 50))
                pixels = list(resized.getdata())
                # Calculate color variance
                r_values = [p[0] for p in pixels]
                g_values = [p[1] for p in pixels]
                b_values = [p[2] for p in pixels]
                
                # High variance in all channels suggests a photo
                r_variance = sum((x - sum(r_values)/len(r_values))**2 for x in r_values) / len(r_values)
                g_variance = sum((x - sum(g_values)/len(g_values))**2 for x in g_values) / len(g_values)
                b_variance = sum((x - sum(b_values)/len(b_values))**2 for x in b_values) / len(b_values)
                
                # If there's significant variance in all channels, it's likely a photo
                is_photo = (r_variance > 100) and (g_variance > 100) and (b_variance > 100)
            except Exception:
                # If analysis fails, don't force JPEG
                is_photo = False
        
        # Determine format - force JPEG for photos if requested and detected
        final_format = "JPEG" if is_photo and force_jpeg_for_photos else output_format
        if is_photo and force_jpeg_for_photos and output_format != "JPEG":
            print(f"Detected photographic image, using JPEG instead of {output_format} for better compression")
        
        # Handle RGBA images again if format changed to JPEG
        if img.mode == 'RGBA' and final_format.upper() in ['JPEG', 'JPG']:
            # Create a white background
            background = Image.new('RGB', img.size, (255, 255, 255))
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
            "WEBP": ".webp"
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
        
        # Initialize attempt counter
        attempts = 0
        current_img = img.copy()
        last_file_size = 0
        
        # Start by using our advanced method if guarantee_size is enabled
        if guarantee_size and not skip_size_check:
            print(f"Using guaranteed size algorithm for {filename}...")
            
            # Process the image with our advanced algorithm
            image_data = adjust_image_size(current_img, final_format, force_size=True)
            
            # Write the file
            with open(output_path, 'wb') as f:
                f.write(image_data)
            
            # Check file size
            file_size = os.path.getsize(output_path)
            
            if 50*1024 <= file_size <= 100*1024:
                print(f"Successfully converted: {output_filename} (Size: {file_size/1024:.2f} KB)")
                return True
            else:
                print(f"Warning: Advanced algorithm couldn't achieve target size for {filename}.")
                print(f"Current size: {file_size/1024:.2f} KB. Falling back to iterative method...")
                # Continue with iterative method
                os.remove(output_path)
        
        # Keep trying until we get within range or reach max attempts
        while attempts < max_attempts:
            attempts += 1
            
            if attempts > 1:
                print(f"Attempt {attempts}/{max_attempts} for {filename}...")
                
                # Apply increasingly aggressive scaling based on attempts
                if last_file_size > 100*1024:  # Too large
                    # More aggressive scaling for each retry
                    scale_factor = max(0.4, 0.7 ** attempts)  # More aggressive with each attempt
                    
                    # Apply scaling
                    width, height = img.size
                    new_width = max(10, int(width * scale_factor))
                    new_height = max(10, int(height * scale_factor))
                    current_img = img.resize((new_width, new_height), Image.LANCZOS)
                    
                    # For PNG, also reduce colors more aggressively with each attempt
                    if final_format.upper() == 'PNG' and attempts > 1:
                        try:
                            # Reduce colors based on attempt number (fewer colors with more attempts)
                            colors = max(8, 256 // (4 ** (attempts - 1)))
                            if current_img.mode == 'RGBA':
                                alpha = current_img.split()[3]
                                rgb_img = current_img.convert('RGB').quantize(colors=colors).convert('RGB')
                                rgb_img.putalpha(alpha)
                                current_img = rgb_img
                            else:
                                current_img = current_img.quantize(colors=colors).convert('RGB')
                        except Exception:
                            # Continue with unmodified image if quantization fails
                            pass
                elif last_file_size < 50*1024:  # Too small
                    # Scale up
                    scale_factor = min(2.0, 1.2 ** attempts)  # More aggressive with each attempt
                    width, height = img.size
                    new_width = min(5000, int(width * scale_factor))
                    new_height = min(5000, int(height * scale_factor))
                    current_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Process the image
            if skip_size_check:
                # Skip size adjustments if requested
                buffer = io.BytesIO()
                if final_format.upper() in ['JPEG', 'JPG']:
                    current_img.save(buffer, format=final_format, quality=90, optimize=True)
                else:
                    current_img.save(buffer, format=final_format, optimize=True)
                image_data = buffer.getvalue()
            else:
                # Apply force_size on later attempts
                use_force = force_size or attempts > 1
                image_data = adjust_image_size(current_img, final_format, force_size=use_force)
            
            # Write the file
            with open(output_path, 'wb') as f:
                f.write(image_data)
            
            # Check if we're in range
            file_size = os.path.getsize(output_path)
            last_file_size = file_size
            
            if skip_size_check or (50*1024 <= file_size <= 100*1024):
                print(f"Successfully converted: {output_filename} (Size: {file_size/1024:.2f} KB)")
                return True
            
            # If not in range but we have attempts left, delete and try again
            if attempts < max_attempts:
                os.remove(output_path)
                print(f"Size out of range: {file_size/1024:.2f} KB - retrying...")
        
        # For small images that can't reach minimum size after multiple attempts,
        # try our special size forcing technique as a last resort
        if last_file_size < 50*1024:
            print(f"Image is too small after {max_attempts} attempts. Using special techniques to increase file size...")
            # Force the image to meet minimum size
            image_data = force_minimum_size(current_img, final_format, target_min=50*1024)
            
            # Write the file
            with open(output_path, 'wb') as f:
                f.write(image_data)
            
            # Check file size
            file_size = os.path.getsize(output_path)
            
            if file_size >= 50*1024 and file_size <= 100*1024:
                print(f"Successfully adjusted size: {output_filename} (Size: {file_size/1024:.2f} KB)")
                return True
            elif file_size > 100*1024:
                print(f"Size still too large ({file_size/1024:.2f} KB). Using aggressive reduction techniques...")
                image_data = force_maximum_size(current_img, final_format, target_max=100*1024)
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                file_size = os.path.getsize(output_path)
                if file_size <= 100*1024:
                    print(f"Successfully reduced size: {output_filename} (Size: {file_size/1024:.2f} KB)")
                    return True
            else:
                print(f"Warning: Could not reach minimum size even with special techniques.")
        elif last_file_size > 100*1024:
            print(f"Image is too large after {max_attempts} attempts. Using aggressive reduction techniques...")
            # Force the image to meet maximum size
            image_data = force_maximum_size(current_img, final_format, target_max=100*1024)
            
            # Write the file
            with open(output_path, 'wb') as f:
                f.write(image_data)
            
            # Check file size
            file_size = os.path.getsize(output_path)
            
            if file_size <= 100*1024:
                print(f"Successfully reduced size: {output_filename} (Size: {file_size/1024:.2f} KB)")
                return True
        else:
            print(f"Warning: Could not reduce size below maximum even with aggressive techniques.")
        
        # If we've reached max attempts, keep the last result but warn user
        print(f"Warning: Could not adjust {output_filename} to size range (50KB-100KB) after {max_attempts} attempts.")
        print(f"Best result: {file_size/1024:.2f} KB")
        return True

    except Exception as e:
        print(f"Error converting {os.path.basename(input_file)}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert images between formats with size constraints (50KB-100KB)')
    parser.add_argument('input', nargs='+', help='Input image file(s). Accepts wildcards.')
    parser.add_argument('-o', '--output-dir', default=os.path.expanduser("~/Documents"), 
                        help='Output directory (default: ~/Documents)')
    parser.add_argument('-f', '--format', choices=['PNG', 'JPEG', 'BMP', 'TIFF', 'GIF', 'WEBP'], 
                        default='PNG', help='Output format (default: PNG)')
    parser.add_argument('--force-jpeg-for-photos', action='store_true',
                        help='Force JPEG format for photographic images regardless of selected format (better compression)')
    parser.add_argument('--force-size', action='store_true',
                        help='Force aggressive resize to meet size constraints')
    parser.add_argument('--skip-size-check', action='store_true',
                        help='Skip size constraints and just convert formats')
    parser.add_argument('--retry-attempts', type=int, default=5,
                        help='Maximum number of attempts to retry compression (default: 5)')
    parser.add_argument('--guarantee-size', action='store_true',
                        help='Use the most advanced method to guarantee target size range')
    
    args = parser.parse_args()
    
    # Expand file globs in input arguments
    input_files = []
    for pattern in args.input:
        expanded = glob.glob(pattern)
        if expanded:
            input_files.extend(expanded)
        else:
            print(f"Warning: No files match pattern '{pattern}'")
    
    if not input_files:
        print("Error: No input files to process")
        return 1
    
    # Check if output directory exists
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
            print(f"Created output directory: {args.output_dir}")
        except Exception as e:
            print(f"Error: Could not create output directory: {e}")
            return 1
    
    # Process files
    total_files = len(input_files)
    print(f"Converting {total_files} files to {args.format} format...")
    
    success_count = 0
    for input_file in input_files:
        if convert_image(input_file, args.output_dir, args.format, 
                        args.force_jpeg_for_photos, args.force_size, 
                        args.skip_size_check, args.retry_attempts,
                        args.guarantee_size):
            success_count += 1
    
    # Show summary
    print(f"\nConversion complete: {success_count} of {total_files} files successfully converted.")
    print(f"Output directory: {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())