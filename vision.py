import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from diffusers import AutoencoderTiny

class VisionSystem:
    def __init__(self, device="cpu"):
        self.device = device
        print("Loading Vision System (CPU-aware)...")
        
        # Concept Extractor (CLIP)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Semantic Vocabulary for Zero-Shot Classification
        self.vocabulary = [
            "animal", "plant", "building", "landscape", "person", "vehicle", "food", 
            "fractal", "crystal", "fluid", "fire", "smoke", "machine", "robot", "cloud",
            "mountain", "ocean", "forest", "city", "space", "geometric", "abstract",
            "organic", "synthetic", "metallic", "wooden", "glass", "glowing", "dark",
            "bright", "colorful", "monochrome", "pattern", "texture", "smooth", "rough",
            "symetrical", "chaotic", "peaceful", "scary", "alien", "familiar", "ancient",
            "futuristic", "microscopic", "macroscopic", "liquid", "solid", "gaseous"
        ]
        print(f"Loaded semantic vocabulary with {len(self.vocabulary)} terms.")
        
        # Pre-compute text features for vocabulary
        with torch.no_grad():
            text_inputs = self.clip_processor(text=self.vocabulary, return_tensors="pt", padding=True).to(self.device)
            self.text_features = self.clip_model.get_text_features(**text_inputs)
            self.text_features /= self.text_features.norm(p=2, dim=-1, keepdim=True)
        
        # Visual Encoder/Decoder (Tiny VAE)
        self.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(self.device)
        
        # Freeze models
        self.clip_model.eval()
        self.vae.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False
            
    def encode_image_to_latent(self, image: Image.Image) -> torch.Tensor:
        """Converts an image to a VAE latent vector."""
        # TAESD expects image in [0, 1] RGB, shape (B, C, H, W)
        img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        # Scale to [-1, 1] as expected by some VAEs, but TAESD handles [0, 1] natively in its encode method if we use diffusers pipeline, 
        # wait, diffusers VAE expects [-1, 1]. Let's use [-1, 1].
        img_tensor = img_tensor * 2.0 - 1.0
        
        with torch.no_grad():
            output = self.vae.encode(img_tensor)
            # TAESD encode returns AutoencoderTinyOutput which has a 'latents' attribute, not 'latent_dist'
            latent = output.latents
        return latent

    def decode_latent_to_image(self, latent: torch.Tensor) -> Image.Image:
        """Converts a VAE latent vector back to an image."""
        # Hard cap the latent values to prevent VAE C++ convolution overflow
        # VAE latents normally live between -3.0 and 3.0.
        # If a mutation pushes a value too high, the decoder's math explodes and silently kills Python.
        latent = torch.clamp(latent, min=-10.0, max=10.0)
        
        try:
            with torch.no_grad():
                output = self.vae.decode(latent).sample
                
            # Output is in [-1, 1], convert to [0, 255]
            output = (output / 2 + 0.5).clamp(0, 1)
            output = output.cpu().permute(0, 2, 3, 1).numpy()[0]
            output = (output * 255).astype(np.uint8)
            return Image.fromarray(output)
        except Exception as e:
            print(f"[Vision] CRITICAL DECODE ERROR: {e}")
            return Image.new('RGB', (512, 512), color='black')

    def extract_concept(self, image: Image.Image) -> torch.Tensor:
        """Extracts CLIP embedding for Reality Anchor and Concept Graph."""
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.get_image_features(**inputs)
        # Normalize
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
        return embedding

    def extract_concept_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decodes latent and extracts its concept embedding."""
        img = self.decode_latent_to_image(latent)
        return self.extract_concept(img)

    def generate_tags(self, image_embedding: torch.Tensor, top_k: int = 3) -> list:
        """Zero-shot classify the image embedding against the vocabulary to generate semantic tags."""
        with torch.no_grad():
            # image_embedding is already normalized
            # shape: [1, 512], text_features shape: [vocab_size, 512]
            similarity = (100.0 * image_embedding @ self.text_features.T).softmax(dim=-1)
            
        values, indices = similarity[0].topk(top_k)
        tags = [self.vocabulary[idx] for idx in indices]
        return tags

    def _ensure_aesthetic_text_features(self):
        """Lazy-init CLIP text banks for multi-facet aesthetic scoring."""
        if hasattr(self, "_aesthetic_ready") and self._aesthetic_ready:
            return

        self.aesthetic_prompts = {
            "striking": [
                "award winning photograph",
                "dramatic lighting masterpiece",
                "iconic powerful visual image",
            ],
            "harmonious": [
                "balanced elegant composition",
                "serene harmonious scenery",
                "tasteful color harmony",
            ],
            "vivid": [
                "sharp detailed high resolution photo",
                "rich vivid colors and texture",
                "crisp clear professional photo",
            ],
            "composition": [
                "rule of thirds composition",
                "well framed artistic photograph",
                "intentional cinematic framing",
            ],
            "overall": [
                "a beautiful high quality photograph",
                "aesthetically pleasing masterpiece",
                "stunning visually appealing image",
            ],
            "anti": [
                "blurry out of focus photo",
                "heavy jpeg compression artifacts",
                "noisy grainy low quality mess",
                "overexposed washed out garbage image",
            ],
        }

        self._aesthetic_features = {}
        with torch.no_grad():
            for facet, prompts in self.aesthetic_prompts.items():
                inputs = self.clip_processor(
                    text=prompts, return_tensors="pt", padding=True
                ).to(self.device)
                feats = self.clip_model.get_text_features(**inputs)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                # Mean prompt vector for the facet
                self._aesthetic_features[facet] = feats.mean(dim=0, keepdim=True)
                self._aesthetic_features[facet] = (
                    self._aesthetic_features[facet]
                    / self._aesthetic_features[facet].norm(p=2, dim=-1, keepdim=True)
                )
        self._aesthetic_ready = True

    def score_aesthetics(self, image_embedding: torch.Tensor) -> dict:
        """
        Multi-facet CLIP aesthetic scores for Beauty and sibling heads.
        Facets are contrastive vs the anti bank (pos - anti). Absolute CLIP
        cosines to abstract aesthetic words are often near-ties, so relative
        contrast is what matters.
        """
        self._ensure_aesthetic_text_features()
        emb = image_embedding
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        emb = emb.to(self.device)
        emb = emb / (emb.norm(p=2, dim=-1, keepdim=True) + 1e-8)

        scores = {}
        with torch.no_grad():
            anti_sim = float((emb @ self._aesthetic_features["anti"].T).squeeze().item())
            facet_sims = {}
            for facet in ("striking", "harmonious", "vivid", "composition", "overall"):
                facet_sims[facet] = float(
                    (emb @ self._aesthetic_features[facet].T).squeeze().item()
                )

            mean_pos = float(np.mean(list(facet_sims.values())))
            # Anti only bites when sludge language beats beauty language
            scores["anti"] = float(1.0 / (1.0 + np.exp(-12.0 * (anti_sim - mean_pos))))

            for facet, pos_sim in facet_sims.items():
                contrast = pos_sim - anti_sim
                scores[facet] = float(1.0 / (1.0 + np.exp(-12.0 * contrast)))

            scores["overall"] = float(
                max(0.0, min(1.0, scores["overall"] * (1.0 - 0.35 * scores["anti"])))
            )
            # Diagnostic: how decisive was CLIP? (near 0.5 = weak signal)
            scores["clip_confidence"] = float(
                max(0.0, min(1.0, abs(mean_pos - anti_sim) / 0.08))
            )
        return scores

    def tag_confidence_peak(self, image_embedding: torch.Tensor) -> float:
        """How peaked the semantic tag distribution is (clear concept vs mush)."""
        emb = image_embedding
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        with torch.no_grad():
            emb = emb / (emb.norm(p=2, dim=-1, keepdim=True) + 1e-8)
            similarity = (100.0 * emb @ self.text_features.T).softmax(dim=-1)[0]
            top = similarity.topk(3).values
            # High if top tag dominates
            peak = float(top[0] / (top.sum() + 1e-8))
        return float(max(0.0, min(1.0, peak)))
