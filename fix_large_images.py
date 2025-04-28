#!/usr/bin/env python3
import os
import sys
from PIL import Image, ImageOps
import io
import glob
from pathlib import Path

def aggressively_reduce_size(input_file, max_size=100*1024):
    """
    Aggressively reduce the size of an image to ensure it's below the maximum size.
    Uses extreme compression techniques for stubborn large files.
    """
    print(f"Processing {os.path.basename(input_file)}...")
    
    try:
        # Open the image
        img = Image.open(input_file)
        
        # Try different quality settings for JPEG
        for quality in [30, 20, 15, 10, 5, 1]:
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            size = buffer.getbuffer().nbytes
            
            if size <= max_size:
                print(f"  Reduced to {size/1024:.2f}KB with quality={quality}")
                return buffer.getvalue()
        
        # If still too large, try aggressive downscaling
        width, height = img.size
        for scale in [0.5, 0.25, 0.125]:
            new_width = max(50, int(width * scale))
            new_height = max(50, int(height * scale))
            resized = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Try with low quality
            for quality in [20, 10, 5, 1]:
                buffer = io.BytesIO()
                resized.save(buffer, format="JPEG", quality=quality, optimize=True)
                size = buffer.getbuffer().nbytes
                
                if size <= max_size:
                    print(f"  Reduced to {size/1024:.2f}KB with scale={scale:.3f}, quality={quality}")
                    return buffer.getvalue()
        
        # Last resort: tiny thumbnail
        img.thumbnail((100, 100), Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=1, optimize=True)
        print(f"  Used last resort: thumbnail 100x100, quality=1")
        return buffer.getvalue()
        
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_large_images.py [directory_path] [max_size_kb]")
        print("Example: python fix_large_images.py ~/Pictures/converted 100")
        return 1
    
    directory = os.path.expanduser(sys.argv[1])
    max_size_kb = 100
    if len(sys.argv) > 2:
        max_size_kb = int(sys.argv[2])
    
    # Find all JPEG images larger than max_size_kb in the directory
    large_files = []
    for file_path in glob.glob(os.path.join(directory, "*.jpg")):
        size_kb = os.path.getsize(file_path) / 1024
        if size_kb > max_size_kb:
            large_files.append((file_path, size_kb))
    
    large_files.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Found {len(large_files)} files larger than {max_size_kb}KB")
    for file_path, size_kb in large_files:
        print(f"{os.path.basename(file_path)}: {size_kb:.2f}KB")
    
    # Process each large file
    for file_path, size_kb in large_files:
        image_data = aggressively_reduce_size(file_path, max_size=max_size_kb*1024)
        if image_data:
            # Backup the original file
            backup_path = file_path + ".bak"
            os.rename(file_path, backup_path)
            
            # Save the reduced file
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            new_size_kb = os.path.getsize(file_path) / 1024
            print(f"Reduced {os.path.basename(file_path)}: {size_kb:.2f}KB → {new_size_kb:.2f}KB")
            
            # If successfully reduced, remove the backup
            if new_size_kb <= max_size_kb:
                os.remove(backup_path)
    
    # Final report
    print("\nFinal results:")
    still_large = 0
    for file_path in glob.glob(os.path.join(directory, "*.jpg")):
        size_kb = os.path.getsize(file_path) / 1024
        if size_kb > max_size_kb:
            print(f"{os.path.basename(file_path)} still too large: {size_kb:.2f}KB")
            still_large += 1
    
    if still_large == 0:
        print("All files successfully reduced below the maximum size!")
    else:
        print(f"{still_large} files still exceed the maximum size.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 