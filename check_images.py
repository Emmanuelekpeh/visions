import os
from PIL import Image

def check_images(directory):
    corrupted = []
    total = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
                total += 1
                filepath = os.path.join(root, file)
                try:
                    with Image.open(filepath) as img:
                        img.verify()
                except Exception as e:
                    corrupted.append(filepath)
    
    print(f"Total images checked: {total}")
    print(f"Corrupted images found: {len(corrupted)}")
    if corrupted:
        print("Sample of corrupted images:")
        for c in corrupted[:5]:
            print(f" - {c}")
            
if __name__ == "__main__":
    check_images("dataset")
