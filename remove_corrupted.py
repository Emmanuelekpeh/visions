import os
from PIL import Image

def remove_corrupted(directory):
    """Remove images that fail verify() or load() (truncated/corrupt downloads)."""
    corrupted = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
                filepath = os.path.join(root, file)
                try:
                    with Image.open(filepath) as img:
                        img.verify()
                    with Image.open(filepath) as img:
                        img.load()
                except Exception:
                    corrupted.append(filepath)

    for c in corrupted:
        try:
            os.remove(c)
            print(f"Removed: {c}")
        except Exception as e:
            print(f"Failed to remove {c}: {e}")

    print(f"Removed {len(corrupted)} corrupted images.")

if __name__ == "__main__":
    remove_corrupted("dataset")
