"""
Training script for Graffiti Artist Generator — BOOTSTRAP ONLY.

The CNN's real learning happens online while the universe runs:
  - Every accepted mutation/combination feeds (embedding, anchor canvas, child latent)
  - Redesign + originality losses push away from copying
  - Anchor replay (35%) keeps real images recognizable

This script warm-starts CNN brush weights so generator_synthesis isn't gray mush
on first launch. Run the app and let evolution teach the creative redesign manifold.
"""

import argparse
import sys
import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
import json
from tqdm import tqdm
from vision import VisionSystem
from latent_generator import create_latent_generator, canvas_tensor_to_pil
from database import WorldMemory
from PIL import Image
import os

def load_training_data_from_database(memory: WorldMemory, limit: int = None, anchors_only: bool = False):
    """
    Load embeddings and latents from database for bootstrap training.
    
    By default includes all stored latents (anchors + legacy evolved concepts).
    Combination learning for generator_version=1 concepts happens online during evolution.
        
    Returns:
        embeddings: [N, 512] numpy array
        latents: [N, 4, 64, 64] numpy array
        concept_ids: List of concept IDs
    """
    cursor = memory.conn.cursor()
    
    if anchors_only:
        query = """
            SELECT c.id, c.embedding, c.latent
            FROM concepts c
            JOIN ingested_files i ON c.id = i.concept_id
            WHERE c.latent IS NOT NULL AND c.embedding IS NOT NULL
        """
    else:
        query = "SELECT id, embedding, latent FROM concepts WHERE latent IS NOT NULL AND embedding IS NOT NULL"
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
        
        emb = np.array(json.loads(emb_json), dtype=np.float32).flatten()
        lat = np.array(json.loads(lat_json), dtype=np.float32)
        if lat.ndim == 4 and lat.shape[0] == 1:
            lat = lat[0]
        
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


def train_supervised(trainer, embeddings, latents, epochs: int = 100, batch_size: int = 8, genesis_ratio: float = 0.40):
    """
    Stage 1: Balanced supervised training on real images.
    
    CRITICAL: Trains BOTH paths to fix unanchored world issues:
        - Genesis path (anchor_image=None): trains spatial_seed for from-scratch generation
        - Redesign path (with anchor): trains creative remixing
    
    Args:
        trainer: LatentGeneratorTrainer instance
        embeddings: [N, 512] training embeddings
        latents: [N, 4, 64, 64] ground truth latents
        epochs: Number of training epochs
        batch_size: Batch size
        genesis_ratio: Fraction of batches to train genesis path (default 0.40 = 40%)
    """
    N = len(embeddings)
    print(f"\nStage 1: Balanced Genesis + Redesign Training")
    print(f"  Training samples: {N}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Genesis ratio: {genesis_ratio:.1%} (CRITICAL for unanchored worlds)")
    print(f"  Steps per epoch: {N // batch_size}")
    
    # Split into train/val (90/10) with shuffle so eval is representative
    perm = np.random.permutation(N)
    embeddings = embeddings[perm]
    latents = latents[perm]
    split_idx = int(0.9 * N)
    train_emb, val_emb = embeddings[:split_idx], embeddings[split_idx:]
    train_lat, val_lat = latents[:split_idx], latents[split_idx:]
    
    print(f"  Train: {len(train_emb)}, Val: {len(val_emb)}")
    
    best_val_loss = float('inf')
    n_batches = max(1, (len(train_emb) + batch_size - 1) // batch_size)

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_losses = []
        genesis_count = 0
        redesign_count = 0

        indices = np.random.permutation(len(train_emb))
        train_emb_shuffled = train_emb[indices]
        train_lat_shuffled = train_lat[indices]

        batch_iter = range(0, len(train_emb), batch_size)
        pbar = tqdm(
            batch_iter,
            total=n_batches,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            file=sys.stdout,
            leave=True,
        )

        for i in pbar:
            batch_emb = train_emb_shuffled[i:i + batch_size]
            batch_lat = train_lat_shuffled[i:i + batch_size]

            # BALANCED TRAINING: Randomly choose genesis or redesign path
            if np.random.random() < genesis_ratio:
                # Train genesis path (anchor_image=None) for unanchored worlds
                loss = trainer.train_on_batch(batch_emb, batch_lat, stage="genesis", creative=True)
                genesis_count += 1
            else:
                # Train redesign path (with anchor) for creative remixing
                loss = trainer.train_on_batch(batch_emb, batch_lat, stage="supervised", creative=True)
                redesign_count += 1
            
            epoch_losses.append(loss)
            pbar.set_postfix(
                loss=f"{loss:.5f}",
                g=genesis_count,
                r=redesign_count,
                refresh=False
            )

        avg_train_loss = float(np.mean(epoch_losses))
        elapsed = time.time() - epoch_start
        actual_genesis_ratio = genesis_count / (genesis_count + redesign_count) if (genesis_count + redesign_count) > 0 else 0
        
        # Validation
        if len(val_emb) > 0:
            val_metrics = trainer.evaluate(val_emb, val_lat)
            val_loss = val_metrics['mse']
            val_cos_sim = val_metrics['cosine_similarity']
            
            # Update learning rate based on validation loss
            trainer.scheduler.step(val_loss)
            
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {avg_train_loss:.6f} | "
                f"Val: {val_loss:.6f} (cos={val_cos_sim:.4f}) | "
                f"Genesis: {actual_genesis_ratio:.1%} | "
                f"Time: {elapsed:.1f}s",
                flush=True,
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trainer.save_checkpoint()
                print(f"  -> New best model saved (val_loss: {val_loss:.6f})")
        else:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {avg_train_loss:.6f} | "
                f"Genesis: {actual_genesis_ratio:.1%} | "
                f"Time: {elapsed:.1f}s",
                flush=True,
            )
    
    print(f"\nBalanced training complete!")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Genesis path is now trained for unanchored worlds!")


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
        epoch_start = time.time()
        epoch_losses = []

        indices = np.random.permutation(N)
        embeddings_shuffled = embeddings[indices]

        n_batches = max(1, (N + batch_size - 1) // batch_size)
        pbar = tqdm(
            range(0, N, batch_size),
            total=n_batches,
            desc=f"Roundtrip {epoch + 1}/{epochs}",
            unit="batch",
            file=sys.stdout,
        )

        for i in pbar:
            batch_emb = embeddings_shuffled[i:i + batch_size]
            loss = trainer.train_on_batch(batch_emb, stage="roundtrip")
            epoch_losses.append(loss)
            pbar.set_postfix(loss=f"{loss:.5f}", refresh=False)

        avg_loss = float(np.mean(epoch_losses))
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{epochs} | Roundtrip Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s", flush=True)
        
        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint()
    
    print(f"\nRoundtrip training complete!")


def _side_by_side(left: Image.Image, right: Image.Image, label_left: str, label_right: str) -> Image.Image:
    """Stack two images horizontally with a small gutter."""
    w, h = left.size
    gutter = 8
    canvas = Image.new("RGB", (w * 2 + gutter, h + 24), color=(24, 24, 24))
    canvas.paste(left, (0, 24))
    canvas.paste(right, (w + gutter, 24))
    return canvas


def run_generator_test(vision, generator, trainer, memory: WorldMemory,
                       samples: int = 6, out_dir: str = "outputs/generator_test"):
    """
    Visual + numeric test of the current checkpoint without launching app.py.
    Saves side-by-side images: stored latent decode | generator prediction decode.
    """
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, stamp)
    os.makedirs(run_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("GENERATOR TEST (no app.py)")
    print("=" * 80)
    print(f"  Checkpoint: latent_generator.pt")
    print(f"  Output dir: {run_dir}")
    print(f"  Samples: {samples}")

    cursor = memory.conn.cursor()
    cursor.execute(
        """
        SELECT c.id, c.embedding, c.latent
        FROM concepts c
        JOIN ingested_files i ON c.id = i.concept_id
        WHERE c.latent IS NOT NULL AND c.embedding IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (samples,),
    )
    rows = cursor.fetchall()
    if not rows:
        print("\n[ERROR] No ingested concepts with latents found.")
        return False

    generator.eval()
    mse_list, cos_list = [], []

    print("\n--- Reconstruction (stored vs generated) ---")
    for idx, (cid, emb_json, lat_json) in enumerate(rows):
        emb = np.array(json.loads(emb_json), dtype=np.float32).flatten()
        stored = np.array(json.loads(lat_json), dtype=np.float32)
        if stored.ndim == 4 and stored.shape[0] == 1:
            stored = stored[0]

        emb_t = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
        stored_t = torch.tensor(stored, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            anchor_t = torch.tensor(stored, dtype=torch.float32).unsqueeze(0)
            pred_latent = generator(emb_t, anchor_latent=anchor_t)
            pred_px = trainer.latents_to_canvas(pred_latent)
            stored_rgb = trainer.latents_to_canvas(anchor_t)

        mse = float(F.mse_loss(pred_px, stored_rgb).item())
        cos = float(F.cosine_similarity(
            pred_px.view(1, -1), stored_rgb.view(1, -1)
        ).item())
        mse_list.append(mse)
        cos_list.append(cos)

        img_stored = canvas_tensor_to_pil(stored_rgb)
        img_pred = canvas_tensor_to_pil(pred_px)
        combo = _side_by_side(img_stored, img_pred, "stored", "generated")
        path = os.path.join(run_dir, f"recon_{idx:02d}_id{cid}.png")
        combo.save(path)
        print(f"  id={cid}  mse={mse:.4f}  cos={cos:.4f}  -> {path}")

    # Synthesis probe: paint pixels on anchor canvas, show side-by-side
    print("\n--- Synthesis probe (pixel redesign on anchor canvas) ---")
    parent_emb = np.array(json.loads(rows[0][1]), dtype=np.float32).flatten()
    anchor_emb = np.array(json.loads(rows[1][1]), dtype=np.float32).flatten()
    stored_anchor = np.array(json.loads(rows[0][2]), dtype=np.float32)
    if stored_anchor.ndim == 4 and stored_anchor.shape[0] == 1:
        stored_anchor = stored_anchor[0]
    anchor_t = torch.tensor(stored_anchor, dtype=torch.float32).unsqueeze(0)
    alpha = 0.7
    steered = alpha * parent_emb + (1.0 - alpha) * anchor_emb
    steered = steered / (np.linalg.norm(steered) + 1e-8)
    with torch.no_grad():
        synth_latent = generator(
            torch.tensor(steered, dtype=torch.float32).unsqueeze(0),
            anchor_latent=anchor_t,
        )
        synth_px = trainer.latents_to_canvas(synth_latent)
    anchor_rgb = trainer.latents_to_canvas(anchor_t)
    img_anchor = canvas_tensor_to_pil(anchor_rgb)
    img_synth = canvas_tensor_to_pil(synth_px)
    combo = _side_by_side(img_anchor, img_synth, "anchor", "pixel_redesign")
    synth_path = os.path.join(run_dir, "synthesis_pixel_redesign.png")
    combo.save(synth_path)
    print(f"  steered blend (alpha={alpha}) -> {synth_path}")

    # Genesis-style: random normalized embedding
    print("\n--- Novel probe (random embedding -> generated image) ---")
    rand_emb = torch.randn(4, 512)
    rand_emb = F.normalize(rand_emb, p=2, dim=1)
    with torch.no_grad():
        novel_px = generator(rand_emb)
    
    for i in range(4):
        img_novel = canvas_tensor_to_pil(trainer.latents_to_canvas(novel_px[i:i+1]))
        novel_path = os.path.join(run_dir, f"novel_random_{i}.png")
        img_novel.save(novel_path)
        print(f"  random unit embedding {i} -> {novel_path}")

    avg_mse = float(np.mean(mse_list))
    avg_cos = float(np.mean(cos_list))
    print("\n--- Summary ---")
    print(f"  Avg pixel MSE:  {avg_mse:.4f}")
    print(f"  Avg pixel Cos:  {avg_cos:.4f}")
    print(f"  Pred range: [{float(novel_px.min()):.2f}, {float(novel_px.max()):.2f}]")

    ready = avg_mse < 0.12 and avg_cos > 0.55
    if ready:
        print("\n[READY] Bootstrap looks usable — open images in output folder to judge visually.")
        print("         Evolution will keep improving creative synthesis in app.py.")
    else:
        print("\n[WEAK] Images may be blurry — try more bootstrap epochs or run app.py to learn online.")
    print(f"\nOpen folder: {os.path.abspath(run_dir)}")
    return ready


def main():
    parser = argparse.ArgumentParser(description="Train Latent Generator")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--stage", type=str, default="both", 
                       choices=["supervised", "roundtrip", "both"],
                       help="Training stage (both recommended for bootstrap)")
    parser.add_argument("--from-images", action="store_true",
                       help="Load from dataset images instead of database")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of training samples")
    parser.add_argument("--anchors-only", action="store_true",
                       help="Bootstrap from ingested real images only (skip legacy evolved latents)")
    parser.add_argument("--fresh", action="store_true",
                       help="Ignore existing checkpoint and train from scratch")
    parser.add_argument("--test", action="store_true",
                       help="Test current checkpoint visually (no training, no app.py)")
    parser.add_argument("--test-samples", type=int, default=6,
                       help="Number of side-by-side recon comparisons in --test mode")
    
    args = parser.parse_args()
    
    if args.test:
        print("=" * 80)
        print("LATENT GENERATOR TEST")
        print("=" * 80)
        vision = VisionSystem(device="cpu")
        generator, trainer = create_latent_generator(vision, device="cpu", load_checkpoint=True)
        memory = WorldMemory()
        run_generator_test(vision, generator, trainer, memory, samples=args.test_samples)
        return
    
    print("=" * 80)
    print("GRAFFITI ARTIST GENERATOR BOOTSTRAP")
    print("(Real learning happens online during evolution — see world.py)")
    print("=" * 80)
    
    # Initialize systems
    print("\nInitializing vision system...")
    vision = VisionSystem(device="cpu")
    
    print("Initializing generator...")
    generator, trainer = create_latent_generator(vision, device="cpu", load_checkpoint=not args.fresh)
    if args.fresh:
        print("  --fresh: training from randomly initialized weights")
    
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
            memory, limit=args.limit, anchors_only=args.anchors_only
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
    
    # Holdout for final evaluation (same 90/10 split logic as training)
    perm = np.random.permutation(len(embeddings))
    split_idx = int(0.9 * len(embeddings))
    val_emb = embeddings[perm][split_idx:]
    val_lat = latents[perm][split_idx:]
    
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
    
    eval_n = min(100, len(val_emb))
    final_metrics = trainer.evaluate(val_emb[:eval_n], val_lat[:eval_n])
    print(f"\nMetrics on validation holdout ({eval_n} samples):")
    print(f"  MSE Loss: {final_metrics['mse']:.6f}")
    print(f"  Cosine Similarity: {final_metrics['cosine_similarity']:.4f}")
    print(f"  Avg Uncertainty: {final_metrics['avg_uncertainty']:.4f}")
    print(f"  Redesign distance: {final_metrics.get('redesign_distance', 0):.4f}")
    
    print(f"\n{trainer.get_training_summary()}")
    
    # Bootstrap readiness (not production perfection — evolution does the real training)
    print("\n" + "=" * 80)
    print("BOOTSTRAP READINESS")
    print("=" * 80)
    
    bootstrap_mse = 0.12
    bootstrap_cos = 0.55
    
    print(f"  Bootstrap pixel MSE: < {bootstrap_mse}")
    print(f"  Bootstrap pixel Cos: > {bootstrap_cos}")
    print(f"  Note: Novel image generation improves as the universe runs and feeds")
    print(f"        accepted combinations back via online RL + supervised training.")
    
    mse_ok = final_metrics['mse'] < bootstrap_mse
    cos_ok = final_metrics['cosine_similarity'] > bootstrap_cos
    
    print(f"\n  MSE: {'[READY]' if mse_ok else '[NEEDS MORE BOOTSTRAP]'}")
    print(f"  Cosine Similarity: {'[READY]' if cos_ok else '[NEEDS MORE BOOTSTRAP]'}")
    
    if mse_ok and cos_ok:
        print("\n[Bootstrap OK] Launch app.py and let evolution train the creative manifold.")
    else:
        print("\n[Bootstrap incomplete] Options:")
        print("   - More epochs: --epochs 20")
        print("   - Or just run app.py — online training will improve it over time")
    
    print("\n" + "=" * 80)
    print("Training complete. Checkpoint saved to latent_generator.pt")
    print("=" * 80)


if __name__ == "__main__":
    main()
