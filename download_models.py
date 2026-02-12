#!/usr/bin/env python3
"""
Model Download Script
====================
Automatically downloads required YOLOv4 model files
"""

import urllib.request
import sys
from pathlib import Path


def download_file(url, destination, description):
    """Download a file with progress indication"""
    print(f"\n📥 Downloading {description}...")
    print(f"   URL: {url}")
    print(f"   Destination: {destination}")
    
    try:
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
            
            # Calculate sizes in MB
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            
            # Print progress bar
            bar_length = 40
            filled = int(bar_length * percent / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f'\r   [{bar}] {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)', 
                  end='', flush=True)
        
        urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        print()  # New line after progress bar
        print(f"   ✅ Successfully downloaded!")
        return True
        
    except Exception as e:
        print(f"\n   ❌ Error downloading file: {e}")
        return False


def main():
    """Main download function"""
    print("=" * 70)
    print("YOLOv4 Model Files Downloader")
    print("=" * 70)
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"\n📁 Models directory: {models_dir.absolute()}")
    
    # Define files to download
    files_to_download = [
        {
            "filename": "coco.names",
            "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
            "description": "COCO class names file"
        },
        {
            "filename": "yolov4.cfg",
            "url": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
            "description": "YOLOv4 configuration file"
        },
        {
            "filename": "yolov4.weights",
            "url": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights",
            "description": "YOLOv4 pre-trained weights (~250 MB)"
        }
    ]
    
    # Download each file
    success_count = 0
    total_files = len(files_to_download)
    
    for idx, file_info in enumerate(files_to_download, 1):
        filepath = models_dir / file_info["filename"]
        
        print(f"\n{'=' * 70}")
        print(f"File {idx}/{total_files}: {file_info['filename']}")
        print(f"{'=' * 70}")
        
        # Check if file already exists
        if filepath.exists():
            print(f"⚠️  File already exists: {filepath}")
            response = input("   Do you want to re-download? (y/N): ").strip().lower()
            
            if response != 'y':
                print("   ⏭️  Skipping...")
                success_count += 1
                continue
        
        # Download the file
        if download_file(file_info["url"], filepath, file_info["description"]):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"✅ Successfully downloaded: {success_count}/{total_files} files")
    
    if success_count == total_files:
        print("\n🎉 All model files are ready!")
        print("\n📝 Next steps:")
        print("   1. Run: python main.py")
        print("   2. Or with options: python main.py --confidence 0.6 --save-output")
    else:
        print(f"\n⚠️  {total_files - success_count} file(s) failed to download")
        print("   Please try downloading them manually (see README.md)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)