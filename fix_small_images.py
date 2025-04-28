#!/usr/bin/env python3
import os
import sys
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageDraw
import io
import glob
import numpy as np
from pathlib import Path

def force_minimum_size(input_file, min_size=50*1024):
    """
    Force an image to meet the minimum file size requirement using various techniques.
    This is specifically for images that are too small to meet the 50KB minimum.
    """
    print(f"Processing {os.path.basename(input_file)}...")
    
    try:
        # Open the image
        img = Image.open(input_file)
        width, height = img.size
        
        # Try these techniques in order of increasing visual impact
        
        # 1. Try maximum quality settings with no optimization
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=100, optimize=False, subsampling=0)
        size = buffer.getbuffer().nbytes
        
        if size >= min_size:
            print(f"  Increased to {size/1024:.2f}KB with max quality")
            return buffer.getvalue()
        
        # 2. Aggressive upscaling for small images
        if width < 1000 or height < 1000:
            scale_factor = max(3.0, 1000 / min(width, height))
            new_width = min(5000, int(width * scale_factor))
            new_height = min(5000, int(height * scale_factor))
            upscaled = img.resize((new_width, new_height), Image.LANCZOS)
            
            buffer = io.BytesIO()
            upscaled.save(buffer, format="JPEG", quality=100, optimize=False)
            size = buffer.getbuffer().nbytes
            
            if size >= min_size:
                print(f"  Increased to {size/1024:.2f}KB with upscaling {scale_factor:.2f}x")
                return buffer.getvalue()
            
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
            noisy_img.save(buffer, format="JPEG", quality=100, optimize=False)
            size = buffer.getbuffer().nbytes
            
            if size >= min_size:
                print(f"  Increased to {size/1024:.2f}KB with noise level {noise_level}")
                return buffer.getvalue()
            
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
            background.save(buffer, format="JPEG", quality=100, optimize=False)
            size = buffer.getbuffer().nbytes
            
            if size >= min_size:
                print(f"  Increased to {size/1024:.2f}KB with {border_size}px border")
                return buffer.getvalue()
        
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
        composite.save(buffer, format="JPEG", quality=100, optimize=False)
        size = buffer.getbuffer().nbytes
        
        if size >= min_size:
            print(f"  Increased to {size/1024:.2f}KB with composite image")
            return buffer.getvalue()
        
        # 6. Final desperate attempt: Add a lot of metadata
        buffer = io.BytesIO()
        composite.save(buffer, format="JPEG", quality=100, optimize=False, 
                     subsampling=0, exif=b"Exif\x00\x00" + bytes([0] * 10000))
        
        print(f"  Used last resort: added metadata padding")
        return buffer.getvalue()
        
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_small_images.py [directory_path] [min_size_kb]")
        print("Example: python fix_small_images.py ~/Pictures/converted 50")
        return 1
    
    directory = os.path.expanduser(sys.argv[1])
    min_size_kb = 50
    if len(sys.argv) > 2:
        min_size_kb = int(sys.argv[2])
    
    # Find all JPEG images smaller than min_size_kb in the directory
    small_files = []
    for file_path in glob.glob(os.path.join(directory, "*.jpg")):
        size_kb = os.path.getsize(file_path) / 1024
        if size_kb < min_size_kb:
            small_files.append((file_path, size_kb))
    
    small_files.sort(key=lambda x: x[1])
    
    print(f"Found {len(small_files)} files smaller than {min_size_kb}KB")
    for file_path, size_kb in small_files:
        print(f"{os.path.basename(file_path)}: {size_kb:.2f}KB")
    
    # Process each small file
    for file_path, size_kb in small_files:
        image_data = force_minimum_size(file_path, min_size=min_size_kb*1024)
        if image_data:
            # Backup the original file
            backup_path = file_path + ".bak"
            os.rename(file_path, backup_path)
            
            # Save the enlarged file
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            new_size_kb = os.path.getsize(file_path) / 1024
            print(f"Enlarged {os.path.basename(file_path)}: {size_kb:.2f}KB → {new_size_kb:.2f}KB")
            
            # If successfully enlarged, remove the backup
            if new_size_kb >= min_size_kb:
                os.remove(backup_path)
    
    # Final report
    print("\nFinal results:")
    still_small = 0
    for file_path in glob.glob(os.path.join(directory, "*.jpg")):
        size_kb = os.path.getsize(file_path) / 1024
        if size_kb < min_size_kb:
            print(f"{os.path.basename(file_path)} still too small: {size_kb:.2f}KB")
            still_small += 1
    
    if still_small == 0:
        print("All files successfully enlarged above the minimum size!")
    else:
        print(f"{still_small} files still below the minimum size.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 