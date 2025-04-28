#!/usr/bin/env python3
import os
import sys
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageDraw
import io
import glob
import numpy as np
import argparse
from pathlib import Path

def force_minimum_size(img, format_name, min_size=50*1024):
    """
    Force an image to meet the minimum file size requirement using various techniques.
    This is specifically for images that are too small to meet the 50KB minimum.
    """
    width, height = img.size
    
    # Try these techniques in order of increasing visual impact
    
    # 1. Try maximum quality settings with no optimization
    buffer = io.BytesIO()
    img.save(buffer, format=format_name, quality=100, optimize=False, subsampling=0)
    size = buffer.getbuffer().nbytes
    
    if size >= min_size:
        return buffer.getvalue(), "max quality"
    
    # 2. Aggressive upscaling for small images
    if width < 1000 or height < 1000:
        scale_factor = max(3.0, 1000 / min(width, height))
        new_width = min(5000, int(width * scale_factor))
        new_height = min(5000, int(height * scale_factor))
        upscaled = img.resize((new_width, new_height), Image.LANCZOS)
        
        buffer = io.BytesIO()
        upscaled.save(buffer, format=format_name, quality=100, optimize=False)
        size = buffer.getbuffer().nbytes
        
        if size >= min_size:
            return buffer.getvalue(), f"upscaling {scale_factor:.2f}x"
        
        img = upscaled  # Continue with the upscaled image
    
    # 3. Add noise to increase entropy and file size
    img_array = np.array(img)
    noise_level = 5  # Start with a small amount
    
    for i in range(3):  # Try increasing noise levels
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
        
        buffer = io.BytesIO()
        noisy_img.save(buffer, format=format_name, quality=100, optimize=False)
        size = buffer.getbuffer().nbytes
        
        if size >= min_size:
            return buffer.getvalue(), f"noise level {noise_level}"
        
        noise_level *= 2  # Double the noise for next attempt
    
    # 4. Add padding/border to increase dimensions
    for border_size in [50, 100, 200, 400]:
        new_width = width + 2 * border_size
        new_height = height + 2 * border_size
        
        # Create background
        background = Image.new('RGB', (new_width, new_height), color=(240, 240, 240))
        draw = ImageDraw.Draw(background)
        
        # Paste the original image in the center
        paste_position = ((new_width - width) // 2, (new_height - height) // 2)
        background.paste(img, paste_position)
        
        buffer = io.BytesIO()
        background.save(buffer, format=format_name, quality=100, optimize=False)
        size = buffer.getbuffer().nbytes
        
        if size >= min_size:
            return buffer.getvalue(), f"{border_size}px border"
    
    # 5. Create a larger composite image with duplicated content
    composite_width = width * 2
    composite_height = height * 2
    composite = Image.new(img.mode, (composite_width, composite_height))
    
    # Paste the image in all four corners with slight variations
    composite.paste(img, (0, 0))
    
    # Apply slight modifications to copies
    modified1 = ImageEnhance.Brightness(img).enhance(1.05)
    modified2 = ImageEnhance.Contrast(img).enhance(1.05)
    modified3 = img.filter(ImageFilter.SHARPEN)
    
    composite.paste(modified1, (width, 0))
    composite.paste(modified2, (0, height))
    composite.paste(modified3, (width, height))
    
    buffer = io.BytesIO()
    composite.save(buffer, format=format_name, quality=100, optimize=False)
    size = buffer.getbuffer().nbytes
    
    if size >= min_size:
        return buffer.getvalue(), "composite image"
    
    # 6. Final desperate attempt: Add a lot of metadata
    buffer = io.BytesIO()
    composite.save(buffer, format=format_name, quality=100, optimize=False, 
                 subsampling=0, exif=b"Exif\x00\x00" + bytes([0] * 10000))
    
    return buffer.getvalue(), "metadata padding"

def force_maximum_size(img, format_name, max_size=100*1024):
    """
    Force an image to stay under the maximum file size requirement using aggressive techniques.
    This is specifically for images that are stubbornly large despite other attempts.
    """
    width, height = img.size
    
    # Start with extreme quality reduction for JPEG
    if format_name.upper() in ['JPEG', 'JPG']:
        for quality in [30, 20, 10, 5, 1]:
            buffer = io.BytesIO()
            img.save(buffer, format=format_name, quality=quality, optimize=True)
            size = buffer.getbuffer().nbytes
            
            if size <= max_size:
                return buffer.getvalue(), f"quality {quality}"
    
    # If quality reduction didn't work, try aggressive downscaling
    downscale_factor = 0.5  # Start with 50% reduction
    
    for i in range(5):  # Try increasingly aggressive downscaling
        new_width = max(50, int(width * downscale_factor))
        new_height = max(50, int(height * downscale_factor))
        
        resized = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Try with lowest quality for JPEG
        if format_name.upper() in ['JPEG', 'JPG']:
            for quality in [20, 10, 5, 1]:
                buffer = io.BytesIO()
                resized.save(buffer, format=format_name, quality=quality, optimize=True)
                size = buffer.getbuffer().nbytes
                
                if size <= max_size:
                    return buffer.getvalue(), f"scale {downscale_factor:.3f}x, quality {quality}"
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
                
                if size <= max_size:
                    return buffer.getvalue(), f"scale {downscale_factor:.3f}x, colors=8"
            except Exception:
                # If quantization fails, just try the resized version
                buffer = io.BytesIO()
                resized.save(buffer, format=format_name, optimize=True)
                size = buffer.getbuffer().nbytes
                
                if size <= max_size:
                    return buffer.getvalue(), f"scale {downscale_factor:.3f}x"
        
        # Try even more aggressive reduction
        downscale_factor *= 0.5  # Reduce by another 50%
    
    # Last resort: Create a tiny thumbnail with minimal colors
    thumbnail_size = (100, 100)  # Extremely small size
    img.thumbnail(thumbnail_size, Image.LANCZOS)
    
    if format_name.upper() in ['JPEG', 'JPG']:
        buffer = io.BytesIO()
        img.save(buffer, format=format_name, quality=1, optimize=True)
        return buffer.getvalue(), "100x100 thumbnail, quality=1"
    else:
        try:
            quantized = img.quantize(colors=4).convert(img.mode)
            buffer = io.BytesIO()
            quantized.save(buffer, format=format_name, optimize=True)
            return buffer.getvalue(), "100x100 thumbnail, colors=4"
        except Exception:
            # Final fallback - just save the thumbnail
            buffer = io.BytesIO()
            img.save(buffer, format=format_name, optimize=True)
            return buffer.getvalue(), "100x100 thumbnail"

def adaptive_sizing(img, format_name, min_size=50*1024, max_size=100*1024, target_ratio=0.9):
    """
    Use a target-based approach to find a size within the min-max range.
    This avoids the ping-pong problem where an image goes from too small to too large repeatedly.
    """
    width, height = img.size
    target_size = min_size + (max_size - min_size) * target_ratio  # Target 90% of range by default
    
    # For JPEG, try quality adjustments first
    if format_name.upper() in ['JPEG', 'JPG']:
        # Binary search for quality
        min_quality = 1
        max_quality = 95
        best_quality = 50
        best_data = None
        best_size = 0
        
        # Try a range of qualities to find the closest match
        for _ in range(10):
            buffer = io.BytesIO()
            img.save(buffer, format=format_name, quality=best_quality, optimize=True)
            size = buffer.getbuffer().nbytes
            
            # If we're in range, return immediately
            if min_size <= size <= max_size:
                return buffer.getvalue(), f"quality {best_quality}"
            
            # Keep track of best result so far
            if abs(size - target_size) < abs(best_size - target_size):
                best_data = buffer.getvalue()
                best_size = size
            
            # Adjust quality
            if size > max_size:
                max_quality = best_quality - 1
                best_quality = (min_quality + best_quality) // 2
            elif size < min_size:
                min_quality = best_quality + 1
                best_quality = (best_quality + max_quality) // 2
            
            # If quality range is very small, we've converged
            if max_quality - min_quality <= 1:
                break
    
    # If quality adjustment didn't work, try scaling
    scale_start = 1.0
    if best_size > max_size:
        # Start with downscaling
        min_scale = 0.1
        max_scale = 0.9
    else:
        # Start with upscaling
        min_scale = 1.1
        max_scale = 3.0
    
    best_scale = scale_start
    best_data = None
    best_size = 0
    
    # Binary search for scale
    for _ in range(10):
        scale = (min_scale + max_scale) / 2
        new_width = max(10, int(width * scale))
        new_height = max(10, int(height * scale))
        
        # Skip if dimensions are too extreme
        if new_width < 10 or new_height < 10 or new_width > 5000 or new_height > 5000:
            break
            
        # Resize and try a few quality settings
        resized = img.resize((new_width, new_height), Image.LANCZOS)
        
        for quality in [85, 70, 55, 40, 25, 10]:
            buffer = io.BytesIO()
            if format_name.upper() in ['JPEG', 'JPG']:
                resized.save(buffer, format=format_name, quality=quality, optimize=True)
            else:
                resized.save(buffer, format=format_name, optimize=True)
                
            size = buffer.getbuffer().nbytes
            
            # If we're in range, return immediately
            if min_size <= size <= max_size:
                return buffer.getvalue(), f"scale {scale:.2f}x, quality {quality}"
            
            # Keep track of best result so far
            if abs(size - target_size) < abs(best_size - target_size):
                best_data = buffer.getvalue()
                best_size = size
                best_scale = scale
                
        # Adjust scale range
        if size > max_size:
            max_scale = scale  # Need smaller
        else:
            min_scale = scale  # Need larger
            
        # If scale range is very small, we've converged
        if abs(max_scale - min_scale) < 0.05:
            break
    
    # Try adding a small amount of noise to fine-tune the size
    if best_size < min_size:
        img_array = np.array(img)
        for noise_level in [2, 5, 10]:
            noisy_array = img_array.copy()
            
            # Add random noise
            if len(img_array.shape) == 3:  # Color image
                for channel in range(img_array.shape[2]):
                    noise = np.random.randint(-noise_level, noise_level, img_array.shape[:2])
                    noisy_array[:, :, channel] = np.clip(noisy_array[:, :, channel] + noise, 0, 255)
            
            noisy_img = Image.fromarray(noisy_array.astype('uint8'))
            buffer = io.BytesIO()
            noisy_img.save(buffer, format=format_name, quality=70, optimize=True)
            size = buffer.getbuffer().nbytes
            
            if min_size <= size <= max_size:
                return buffer.getvalue(), f"noise level {noise_level}, quality 70"
    
    # Return the best approximation we found
    if min_size <= best_size <= max_size:
        return best_data, f"best approximation (size: {best_size/1024:.2f}KB)"
    else:
        # Try one final approach - scale to exactly the target size using extreme methods if needed
        if best_size > max_size:
            return force_maximum_size(img, format_name, max_size)
        else:
            return force_minimum_size(img, format_name, min_size)

def ensure_size_range(input_file, min_size_kb=50, max_size_kb=100, format_name="JPEG"):
    """Process a file to ensure it's within the specified size range."""
    print(f"Processing {os.path.basename(input_file)}...")
    
    try:
        # Open the image
        img = Image.open(input_file)
        current_size_kb = os.path.getsize(input_file) / 1024
        
        # Check if it's already in range
        if min_size_kb <= current_size_kb <= max_size_kb:
            print(f"  Already in range: {current_size_kb:.2f}KB")
            return True
        
        # Backup the original file
        backup_path = input_file + ".bak"
        os.rename(input_file, backup_path)
        
        # Use adaptive sizing to avoid ping-pong between too small and too large
        image_data, technique = adaptive_sizing(
            img, format_name, 
            min_size=min_size_kb*1024, 
            max_size=max_size_kb*1024,
            target_ratio=0.75  # Target 75% of the range to avoid hitting max
        )
        
        # Save the adjusted file
        with open(input_file, 'wb') as f:
            f.write(image_data)
        
        new_size_kb = os.path.getsize(input_file) / 1024
        print(f"  Adjusted: {current_size_kb:.2f}KB → {new_size_kb:.2f}KB (using {technique})")
        
        # Check if we succeeded
        if min_size_kb <= new_size_kb <= max_size_kb:
            print(f"  SUCCESS: Final size {new_size_kb:.2f}KB is within range")
            os.remove(backup_path)  # Remove backup if successful
            return True
        else:
            print(f"  WARNING: Final size {new_size_kb:.2f}KB is outside target range")
            # Try one more approach if still outside range
            if new_size_kb > max_size_kb:
                print(f"  Attempting final resize to reduce file size...")
                img_resized = Image.open(input_file)
                # Try a more aggressive approach
                for quality in [30, 20, 10, 5, 1]:
                    buffer = io.BytesIO()
                    img_resized.save(buffer, format=format_name, quality=quality, optimize=True)
                    size = buffer.getbuffer().nbytes
                    if size <= max_size_kb*1024:
                        with open(input_file, 'wb') as f:
                            f.write(buffer.getvalue())
                        final_size_kb = os.path.getsize(input_file) / 1024
                        print(f"  Finally reduced to {final_size_kb:.2f}KB with quality={quality}")
                        if min_size_kb <= final_size_kb <= max_size_kb:
                            os.remove(backup_path)
                            return True
            elif new_size_kb < min_size_kb:
                print(f"  Attempting final resize to increase file size...")
                img_resized = Image.open(input_file)
                # Try a more aggressive approach to increase size
                buffer = io.BytesIO()
                img_resized.save(buffer, format=format_name, quality=100, optimize=False, subsampling=0)
                size = buffer.getbuffer().nbytes
                if size >= min_size_kb*1024:
                    with open(input_file, 'wb') as f:
                        f.write(buffer.getvalue())
                    final_size_kb = os.path.getsize(input_file) / 1024
                    print(f"  Finally increased to {final_size_kb:.2f}KB with max quality")
                    if min_size_kb <= final_size_kb <= max_size_kb:
                        os.remove(backup_path)
                        return True
            
            # Keep both files in case of failure
            os.rename(backup_path, input_file + ".original")
            return False
            
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Ensure image files are within a specific size range (in KB)")
    parser.add_argument("directory", help="Directory containing images to process")
    parser.add_argument("--min", type=int, default=50, help="Minimum file size in KB (default: 50)")
    parser.add_argument("--max", type=int, default=100, help="Maximum file size in KB (default: 100)")
    parser.add_argument("--format", choices=["JPEG", "PNG", "GIF", "WEBP"], default="JPEG", 
                        help="Output format (default: JPEG)")
    parser.add_argument("--extension", default=".jpg", help="File extension to process (default: .jpg)")
    
    args = parser.parse_args()
    
    # Expand user directory if needed
    directory = os.path.expanduser(args.directory)
    
    # Get list of images to process
    files_to_process = []
    for file_path in glob.glob(os.path.join(directory, f"*{args.extension}")):
        size_kb = os.path.getsize(file_path) / 1024
        if not (args.min <= size_kb <= args.max):
            files_to_process.append((file_path, size_kb))
    
    # Sort files by how far they are from the target range
    target_mid = (args.min + args.max) / 2
    files_to_process.sort(key=lambda x: abs(x[1] - target_mid), reverse=True)
    
    # Print summary
    print(f"Found {len(files_to_process)} files outside the target range ({args.min}-{args.max}KB)")
    if files_to_process:
        print("Files outside range:")
        for file_path, size_kb in files_to_process:
            status = "too small" if size_kb < args.min else "too large"
            print(f"  {os.path.basename(file_path)}: {size_kb:.2f}KB ({status})")
        print()
    
    # Process each file
    success_count = 0
    for file_path, _ in files_to_process:
        if ensure_size_range(file_path, args.min, args.max, args.format):
            success_count += 1
        print()  # Add a line break for readability
    
    # Final report
    print("\nFinal results:")
    print(f"Successfully processed {success_count} of {len(files_to_process)} files")
    
    # Check for remaining files outside range
    files_outside_range = 0
    for file_path in glob.glob(os.path.join(directory, f"*{args.extension}")):
        size_kb = os.path.getsize(file_path) / 1024
        if not (args.min <= size_kb <= args.max):
            print(f"{os.path.basename(file_path)} still outside range: {size_kb:.2f}KB")
            files_outside_range += 1
    
    if files_outside_range == 0 and files_to_process:
        print("All files are now within the target size range!")
    elif files_outside_range > 0:
        print(f"{files_outside_range} files still outside the target range.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 