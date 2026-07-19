"""
Training script for Latent Generator
Bootstraps the generator on existing real images in the database.

Usage:
    python train_latent_generator.py [--epochs 100] [--batch-size 8] [--stage supervised]
"""

import argparse
import numpy as np
import torch
import json
from tqdm import tqdm
from vision import VisionSystem
from latent_generator import create_latent_generator
from database import WorldMemory
from PIL import Image
import os

def load_training_data_from_database(memory: WorldMemory, limit: int = None):
    """
    Load embeddings and latents from database for training.
    
    Args:
        memory: WorldMemory instance
        limit: Optional limit on number of samples
        
    Returns:
        embeddings: [N, 512] numpy array
        latents: [N, 4, 64, 64] numpy array
        concept_ids: List of concept IDs
    """
    cursor = memory.conn.cursor()
    
    # Get all concepts with stored latents
    query = "SELECT id, embedding, latent FROM concepts WHERE latent IS NOT NULL"
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    if not rows:
        print("No concepts with stored latents found in database!")
        return None, None, None
    
    embeddings = []
    latents = []
    concept_ids = []
    
    for row in rows:
        concept_id, emb_json, lat_json = row
        
        emb = np.array(json.loads(emb_json), dtype=np.float32)
        lat = np.array(json.loads(lat_json), dtype=np.float32)
        
        embeddings.append(emb)
        latents.append(lat)
        concept_ids.append(concept_id)
    
    embeddings = np.array(embeddings)
    latents = np.array(latents)
    
    return embeddings, latents, concept_ids


def load_training_data_from_images(dataset_path: str, vision: VisionSystem, limit: int = None):
    """
    Load training data directly from image files (if database is empty).
    
    Args:
        dataset_path: Path to dataset folder
        vision: VisionSystem instance
        limit: Optional limit on number of images
        
    Returns:
        embeddings: [N, 512] numpy array
        latents: [N, 4, 64, 64] numpy array
        filenames: List of filenames
    """
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        return None, None, None
    
    # Find all images
    image_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                image_files.append(os.path.join(root, file))
    
    if limit:
        image_files = image_files[:limit]
    
    if not image_files:
        print(f"No images found in {dataset_path}")
        return None, None, None
    
    print(f"Processing {len(image_files)} images from dataset...")
    
    embeddings = []
    latents = []
    filenames = []
    
    for img_path in tqdm(image_files, desc="Loading images"):
        try:
            # Load image
            img = Image.open(img_path).convert("RGB")
            img = img.resize((512, 512))
            
            # Extract embedding
            embedding = vision.extract_concept(img)
            embedding_np = embedding.cpu().numpy().flatten()
            
            # Encode to latent
            latent = vision.encode_image_to_latent(img)
            latent_np = latent.cpu().numpy()
            
            # Remove batch dimension if present (shape should be [4, 64, 64], not [1, 4, 64, 64])
            if latent_np.ndim == 4 and latent_np.shape[0] == 1:
                latent_np = latent_np[0]
            
            embeddings.append(embedding_np)
            latents.append(latent_np)
            filenames.append(os.path.basename(img_path))
            
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
            continue
    
    if not embeddings:
        return None, None, None
    
    embeddings = np.array(embeddings)
    latents = np.array(latents)
    
    return embeddings, latents, filenames


def train_supervised(trainer, embeddings, latents, epochs: int = 100, batch_size: int = 8):
    """
    Stage 1: Supervised training on real images.
    
    Args:
        trainer: LatentGeneratorTrainer instance
        embeddings: [N, 512] training embeddings
        latents: [N, 4, 64, 64] ground truth latents
        epochs: Number of training epochs
        batch_size: Batch size
    """
    N = len(embeddings)
    print(f"\nStage 1: Supervised Training")
    print(f"  Training samples: {N}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps per epoch: {N // batch_size}")
    
    # Split into train/val (90/10)
    split_idx = int(0.9 * N)
    train_emb, val_emb = embeddings[:split_idx], embeddings[split_idx:]
    train_lat, val_lat = latents[:split_idx], latents[split_idx:]
    
    print(f"  Train: {len(train_emb)}, Val: {len(val_emb)}")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        epoch_losses = []
        
        # Shuffle training data
        indices = np.random.permutation(len(train_emb))
        train_emb_shuffled = train_emb[indices]
        train_lat_shuffled = train_lat[indices]
        
        # Mini-batch training
        for i in range(0, len(train_emb), batch_size):
            batch_emb = train_emb_shuffled[i:i+batch_size]
            batch_lat = train_lat_shuffled[i:i+batch_size]
            
            loss = trainer.train_on_batch(batch_emb, batch_lat, stage="supervised")
            epoch_losses.append(loss)
        
        avg_train_loss = np.mean(epoch_losses)
        
        # Validation
        if len(val_emb) > 0:
            val_metrics = trainer.evaluate(val_emb, val_lat)
            val_loss = val_metrics['mse']
            val_cos_sim = val_metrics['cosine_similarity']
            
            # Update learning rate based on validation loss
            trainer.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val Cos Sim: {val_cos_sim:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trainer.save_checkpoint()
                print(f"  -> New best model saved (val_loss: {val_loss:.6f})")
        else:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f}")
    
    print(f"\nSupervised training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")


def train_roundtrip(trainer, embeddings, epochs: int = 50, batch_size: int = 8):
    """
    Stage 2: Self-supervised roundtrip consistency training.
    
    Args:
        trainer: LatentGeneratorTrainer instance
        embeddings: [N, 512] training embeddings
        epochs: Number of training epochs
        batch_size: Batch size
    """
    N = len(embeddings)
    print(f"\nStage 2: Self-Supervised Roundtrip Training")
    print(f"  Training samples: {N}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    
    for epoch in range(epochs):
        epoch_losses = []
        
        # Shuffle
        indices = np.random.permutation(N)
        embeddings_shuffled = embeddings[indices]
        
        # Mini-batch training
        for i in range(0, N, batch_size):
            batch_emb = embeddings_shuffled[i:i+batch_size]
            
            loss = trainer.train_on_batch(batch_emb, stage="roundtrip")
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{epochs} | Roundtrip Loss: {avg_loss:.6f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint()
    
    print(f"\nRoundtrip training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train Latent Generator")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--stage", type=str, default="supervised", 
                       choices=["supervised", "roundtrip", "both"],
                       help="Training stage")
    parser.add_argument("--from-images", action="store_true",
                       help="Load from dataset images instead of database")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of training samples")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LATENT GENERATOR TRAINING")
    print("=" * 80)
    
    # Initialize systems
    print("\nInitializing vision system...")
    vision = VisionSystem(device="cpu")
    
    print("Initializing generator...")
    generator, trainer = create_latent_generator(vision, device="cpu")
    
    # Load training data
    if args.from_images:
        print("\nLoading training data from images...")
        embeddings, latents, ids = load_training_data_from_images(
            "dataset", vision, limit=args.limit
        )
    else:
        print("\nLoading training data from database...")
        memory = WorldMemory()
        embeddings, latents, ids = load_training_data_from_database(
            memory, limit=args.limit
        )
    
    if embeddings is None or len(embeddings) == 0:
        print("\n[ERROR] No training data available!")
        print("Options:")
        print("  1. Use --from-images to load from dataset/ folder")
        print("  2. First run ingest_dataset.py to populate database")
        return
    
    print(f"\nLoaded {len(embeddings)} training samples")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Latent shape: {latents.shape}")
    
    # Train
    if args.stage in ["supervised", "both"]:
        train_supervised(trainer, embeddings, latents, 
                        epochs=args.epochs, batch_size=args.batch_size)
    
    if args.stage in ["roundtrip", "both"]:
        roundtrip_epochs = args.epochs // 2 if args.stage == "both" else args.epochs
        train_roundtrip(trainer, embeddings,
                       epochs=roundtrip_epochs, batch_size=args.batch_size)
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    final_metrics = trainer.evaluate(embeddings[:100], latents[:100])
    print(f"\nMetrics on first 100 samples:")
    print(f"  MSE Loss: {final_metrics['mse']:.6f}")
    print(f"  Cosine Similarity: {final_metrics['cosine_similarity']:.4f}")
    print(f"  Avg Uncertainty: {final_metrics['avg_uncertainty']:.4f}")
    
    print(f"\n{trainer.get_training_summary()}")
    
    # Success criteria
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA")
    print("=" * 80)
    
    target_mse = 0.01
    target_cos_sim = 0.95
    
    print(f"  Target MSE: < {target_mse}")
    print(f"  Target Cosine Similarity: > {target_cos_sim}")
    
    mse_ok = final_metrics['mse'] < target_mse
    cos_ok = final_metrics['cosine_similarity'] > target_cos_sim
    
    print(f"\n  MSE: {'[PASS]' if mse_ok else '[FAIL]'}")
    print(f"  Cosine Similarity: {'[PASS]' if cos_ok else '[FAIL]'}")
    
    if mse_ok and cos_ok:
        print("\n[SUCCESS] Generator ready for production use!")
    else:
        print("\n[WARNING] Generator needs more training or architecture tuning")
        print("   Consider:")
        print("   - More training epochs (--epochs 200)")
        print("   - More training data")
        print("   - Adjusting learning rate")
    
    print("\n" + "=" * 80)
    print("Training complete. Checkpoint saved to latent_generator.pt")
    print("=" * 80)


if __name__ == "__main__":
    main()
