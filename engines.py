import torch
import numpy as np
import random

class DreamEngine:
    def __init__(self, device="cpu"):
        self.device = device
        
    def mutate_latent(self, latent: torch.Tensor, mutation_rate=0.1, temperature=1.0) -> torch.Tensor:
        """Adds noise to the latent to explore nearby possibilities while maintaining variance."""
        noise = torch.randn_like(latent) * temperature
        # Blend latent and noise using spherical interpolation logic to preserve variance
        a = np.sqrt(max(0.0, 1.0 - mutation_rate**2))
        mutated = latent * a + noise * mutation_rate
        return mutated
        
    def recombine_latents(self, latent1: torch.Tensor, latent2: torch.Tensor, alpha=0.5) -> torch.Tensor:
        """Interpolates between two latents."""
        recombined = latent1 * alpha + latent2 * (1.0 - alpha)
        # Restore variance (assuming independent distributions)
        var_factor = np.sqrt(alpha**2 + (1.0 - alpha)**2)
        return recombined / max(1e-4, var_factor)

class CuriosityEngine:
    def __init__(self):
        pass
        
    def calculate_reward(self, evaluation: dict, times_referenced: int) -> float:
        """
        Calculates reward for a concept.
        Reward = novelty + uncertainty + connectivity - repetition
        """
        novelty = evaluation["novelty"]
        # Uncertainty is high when coherence is medium (not too rigid, not pure noise)
        uncertainty = 1.0 - abs(evaluation["coherence"] - 0.5) * 2
        
        # Repetition penalty
        repetition = min(1.0, times_referenced / 10.0)
        
        reward = novelty + uncertainty - repetition
        return reward
