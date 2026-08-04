"""
Multi-Head Reinforcement Learning System
Inspired by biological dopamine/motivation systems.

Each head is multi-faceted: several related signals are fused so the judge
understands Beauty / Novelty / etc. as rich concepts, not single proxies.

Heads share one SharedSignals snapshot (geometry, CLIP aesthetics, reality,
lineage, observer, prediction error). After per-head scoring, a coupling
pass lets sibling heads inform each other so judgments stay coherent.

Meta-Judge:
- Dynamically weights objectives based on urgency, scarcity, fatigue
- Prevents reward hacking through diversity enforcement
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SharedSignals:
    """
    One concept, many measurements. All heads read from this so they share
    the same underlying understanding instead of inventing isolated proxies.
    """
    # Geometry in concept space
    nearest_dist: float = 1.0
    avg_nn_dist: float = 1.0
    nn_variance: float = 0.5
    local_density: float = 0.5  # inverse of avg distance, clipped

    # Reality grounding
    reality_distance: float = 1.0
    reality_affinity: float = 0.5  # high = close to real images
    grounded_novelty: float = 0.5  # novel but not unanchored chaos

    # CLIP aesthetic facets (multi-meaning beauty)
    aesthetic_overall: float = 0.5
    aesthetic_striking: float = 0.5
    aesthetic_harmonious: float = 0.5
    aesthetic_vivid: float = 0.5
    aesthetic_composition: float = 0.5
    anti_aesthetic: float = 0.0  # muddy / blurry / sludge
    clip_confidence: float = 0.0  # how decisive CLIP contrast was

    # Pixel CNN + heuristics (decoded image — what CLIP cannot see)
    pixel_quality: float = 0.5
    pixel_sludge: float = 0.0
    pixel_sharpness: float = 0.5
    pixel_diversity: float = 0.5

    # Semantic structure
    tag_peakiness: float = 0.5
    tags: List[str] = field(default_factory=list)

    # Texture regime: counter crystal/sharp attractor bias
    soft_affinity: float = 0.5       # fluid/organic/smooth tag + low-freq latent
    crystal_affinity: float = 0.5    # fractal/crystal/geometric tag + high-freq latent
    texture_balance: float = 0.5     # high = soft-leaning, low = knife/crystal
    latent_highfreq: float = 0.5     # spatial high-frequency energy in VAE latent

    # Latent / representation
    latent_energy: float = 0.5
    latent_channel_balance: float = 0.5
    embedding_norm: float = 0.5
    embedding_entropy: float = 0.5

    # Lineage
    parent_alignment: float = 0.3
    parent_spread: float = 0.0  # distance between parents if recombination

    # Learning / observer
    prediction_error: float = 0.3
    observer_boost: float = 0.0
    under_explored: float = 0.5


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def build_shared_signals(
    concept_embedding: np.ndarray,
    latent: torch.Tensor,
    memory,
    parent_ids: Optional[List[int]] = None,
    vision=None,
    reality_distance: Optional[float] = None,
    prediction_error: float = 0.3,
    observer_boost: float = 0.0,
    tags: Optional[List[str]] = None,
    aesthetic_scores: Optional[Dict[str, float]] = None,
    pixel_scores: Optional[Dict[str, float]] = None,
    **kwargs,
) -> SharedSignals:
    """Compute the shared measurement snapshot used by every head."""
    emb = np.asarray(concept_embedding, dtype=np.float32).flatten()
    signals = SharedSignals()
    signals.prediction_error = float(prediction_error)
    signals.observer_boost = _clip01(observer_boost)
    signals.tags = list(tags or [])

    # --- Geometry ---
    nearest = []
    if memory is not None and getattr(memory, "index", None) is not None and memory.index.ntotal > 0:
        nearest = memory.get_nearest_concepts(emb, k=min(20, memory.index.ntotal)) or []

    if nearest:
        dists = [float(d) for _, d in nearest]
        signals.nearest_dist = dists[0]
        signals.avg_nn_dist = float(np.mean(dists[:5])) if len(dists) >= 5 else float(np.mean(dists))
        signals.nn_variance = float(np.var(dists)) if len(dists) > 1 else 0.5
        signals.local_density = _clip01(1.0 / (1.0 + signals.avg_nn_dist))
        signals.under_explored = _clip01(signals.nn_variance / 2.0)
    else:
        signals.nearest_dist = 3.0
        signals.avg_nn_dist = 3.0
        signals.nn_variance = 1.0
        signals.local_density = 0.1
        signals.under_explored = 0.9

    # --- Reality ---
    if reality_distance is not None:
        signals.reality_distance = float(reality_distance)
    else:
        signals.reality_distance = signals.nearest_dist  # weak fallback
    # Affinity: close to reality is high; far collapses
    signals.reality_affinity = _clip01(1.0 - signals.reality_distance / 2.5)
    # Grounded novelty: peak when somewhat away from both neighbors AND reality soup
    # Sweet zone ~0.8–1.6 reality distance
    rd = signals.reality_distance
    signals.grounded_novelty = _clip01(1.0 - abs(rd - 1.2) / 1.5)

    # --- Aesthetic (CLIP multi-facet or precomputed) ---
    if aesthetic_scores:
        signals.aesthetic_overall = _clip01(aesthetic_scores.get("overall", 0.5))
        signals.aesthetic_striking = _clip01(aesthetic_scores.get("striking", 0.5))
        signals.aesthetic_harmonious = _clip01(aesthetic_scores.get("harmonious", 0.5))
        signals.aesthetic_vivid = _clip01(aesthetic_scores.get("vivid", 0.5))
        signals.aesthetic_composition = _clip01(aesthetic_scores.get("composition", 0.5))
        signals.anti_aesthetic = _clip01(aesthetic_scores.get("anti", 0.0))
        signals.clip_confidence = _clip01(aesthetic_scores.get("clip_confidence", 0.0))
    elif vision is not None and hasattr(vision, "score_aesthetics"):
        emb_t = torch.tensor(emb, dtype=torch.float32)
        if emb_t.dim() == 1:
            emb_t = emb_t.unsqueeze(0)
        scored = vision.score_aesthetics(emb_t)
        signals.aesthetic_overall = _clip01(scored.get("overall", 0.5))
        signals.aesthetic_striking = _clip01(scored.get("striking", 0.5))
        signals.aesthetic_harmonious = _clip01(scored.get("harmonious", 0.5))
        signals.aesthetic_vivid = _clip01(scored.get("vivid", 0.5))
        signals.aesthetic_composition = _clip01(scored.get("composition", 0.5))
        signals.anti_aesthetic = _clip01(scored.get("anti", 0.0))
        signals.clip_confidence = _clip01(scored.get("clip_confidence", 0.0))

    # --- Pixel eyes (CNN + heuristics on decoded image) ---
    if pixel_scores:
        signals.pixel_quality = _clip01(pixel_scores.get("quality", 0.5))
        signals.pixel_sludge = _clip01(pixel_scores.get("sludge", 0.0))
        signals.pixel_sharpness = _clip01(pixel_scores.get("sharpness", 0.5))
        signals.pixel_diversity = _clip01(pixel_scores.get("diversity", 0.5))
        # Pixel sludge overrides CLIP when they disagree — CLIP is blind to mush
        signals.anti_aesthetic = _clip01(
            max(signals.anti_aesthetic, signals.pixel_sludge * 0.85)
        )

    # --- Semantic peakiness ---
    if vision is not None and hasattr(vision, "tag_confidence_peak") and emb.size > 0:
        signals.tag_peakiness = _clip01(vision.tag_confidence_peak(
            torch.tensor(emb, dtype=torch.float32).unsqueeze(0) if emb.ndim == 1 else torch.tensor(emb)
        ))
    elif signals.tags:
        signals.tag_peakiness = 0.6  # had tags but no distribution
    else:
        signals.tag_peakiness = 0.4

    # --- Latent / embedding structure ---
    latent_np = latent.detach().cpu().numpy().astype(np.float32)
    signals.latent_energy = _clip01(float(np.sqrt(np.mean(latent_np ** 2))) / 2.0)
    # Channel balance: similar energy across VAE channels ≈ structured, not one-channel noise
    if latent_np.ndim >= 2:
        # [B,C,H,W] -> per-channel std
        if latent_np.ndim == 4:
            channel_stds = latent_np[0].std(axis=(1, 2))
            mean_std = float(np.mean(channel_stds) + 1e-8)
            balance = 1.0 - float(np.std(channel_stds) / mean_std)
            signals.latent_channel_balance = _clip01(balance)
            # High-frequency energy: residual after spatial blur
            vol = latent_np[0]  # [C,H,W]
            # Cheap box blur via 2x2 mean of non-overlapping blocks then upsample-ish compare
            c, h, w = vol.shape
            if h >= 4 and w >= 4:
                # Average neighboring pixels as low-pass proxy
                low = vol.copy()
                low[:, 1:-1, 1:-1] = (
                    vol[:, :-2, 1:-1] + vol[:, 2:, 1:-1] +
                    vol[:, 1:-1, :-2] + vol[:, 1:-1, 2:]
                ) * 0.25
                high = vol - low
                signals.latent_highfreq = _clip01(float(np.sqrt(np.mean(high ** 2))) / 1.5)
            else:
                signals.latent_highfreq = 0.5
        else:
            signals.latent_channel_balance = _clip01(1.0 / (1.0 + float(np.std(latent_np))))
            signals.latent_highfreq = 0.5
    else:
        signals.latent_channel_balance = 0.5
        signals.latent_highfreq = 0.5

    signals.embedding_norm = _clip01(float(np.linalg.norm(emb)) / 2.0)
    abs_emb = np.abs(emb) + 1e-8
    abs_emb = abs_emb / abs_emb.sum()
    entropy = -float(np.sum(abs_emb * np.log(abs_emb)))
    signals.embedding_entropy = _clip01(entropy / 6.0)

    # --- Texture regime (soft vs crystal) ---
    SOFT_TAGS = {
        "fluid", "organic", "smoke", "cloud", "smooth", "liquid", "peaceful",
        "landscape", "ocean", "forest", "fire", "gaseous", "plant", "soft",
    }
    SHARP_TAGS = {
        "fractal", "crystal", "geometric", "metallic", "synthetic", "pattern",
        "machine", "robot", "glass", "symetrical", "solid", "microscopic",
    }
    tag_set = {str(t).lower() for t in signals.tags}
    soft_hits = len(tag_set & SOFT_TAGS)
    sharp_hits = len(tag_set & SHARP_TAGS)
    tag_soft = soft_hits / max(1, soft_hits + sharp_hits) if (soft_hits + sharp_hits) else 0.5
    # Latent: low highfreq → soft
    latent_soft = 1.0 - signals.latent_highfreq
    signals.soft_affinity = _clip01(0.55 * tag_soft + 0.45 * latent_soft)
    signals.crystal_affinity = _clip01(0.55 * (1.0 - tag_soft) + 0.45 * signals.latent_highfreq)
    # Balance: prefer soft when crystal would otherwise dominate survivors
    signals.texture_balance = _clip01(
        0.5 + 0.5 * (signals.soft_affinity - signals.crystal_affinity)
    )

    # --- Lineage ---
    parent_ids = parent_ids or []
    parent_distances = []
    parent_embs = []
    if parent_ids and memory is not None:
        import json
        for pid in parent_ids:
            row = memory.get_concept(pid)
            if row:
                parent_emb = np.array(json.loads(row[1]), dtype=np.float32).flatten()
                parent_embs.append(parent_emb)
                parent_distances.append(float(np.linalg.norm(emb - parent_emb)))
    if parent_distances:
        avg_p = float(np.mean(parent_distances))
        signals.parent_alignment = _clip01(1.0 - avg_p / 2.0)
        if len(parent_embs) >= 2:
            signals.parent_spread = _clip01(float(np.linalg.norm(parent_embs[0] - parent_embs[1])) / 3.0)
    else:
        signals.parent_alignment = 0.25

    return signals


class RewardHead:
    """Base class for individual reward objectives."""
    def __init__(self, name: str, base_weight: float = 1.0):
        self.name = name
        self.base_weight = base_weight
        self.history = []
        self.last_activation = 0
        self.total_activations = 0
        self.last_facets: Dict[str, float] = {}

    def evaluate(self, concept_embedding: np.ndarray, latent: torch.Tensor,
                 memory, signals: SharedSignals = None, **kwargs) -> float:
        raise NotImplementedError

    def record_reward(self, reward: float, timestamp: float):
        self.history.append((timestamp, reward))
        if len(self.history) > 100:
            self.history = self.history[-100:]
        if reward > 0.7:
            self.last_activation = timestamp
            self.total_activations += 1


class BeautyHead(RewardHead):
    """
    Beauty is plural: striking, harmonious, vivid, composed, grounded in
    reality's look, cleared of sludge, optionally endorsed by the observer.
    When CLIP contrast is weak, reality resonance + clarity carry more weight.
    """
    def __init__(self):
        super().__init__("Beauty", base_weight=0.25)

    def evaluate(self, concept_embedding, latent, memory, signals: SharedSignals = None, **kwargs) -> float:
        s = signals
        clip_w = 0.25 + 0.25 * s.clip_confidence  # trust CLIP less — pixels matter more
        ground_w = 1.0 - 0.15 * s.clip_confidence

        facets = {
            "striking": s.aesthetic_striking,
            "harmonious": s.aesthetic_harmonious,
            "vivid": s.aesthetic_vivid,
            "composition": s.aesthetic_composition,
            "clip_overall": s.aesthetic_overall,
            "reality_resonance": _clip01(
                s.reality_affinity * (1.0 - 0.35 * abs(s.reality_distance - 0.85) / 2.0)
            ),
            "clarity": _clip01(s.tag_peakiness * (1.0 - s.anti_aesthetic)),
            "structure": _clip01(0.5 * s.latent_channel_balance + 0.5 * s.pixel_sharpness),
            "observer": s.observer_boost,
            "anti_sludge": _clip01(1.0 - max(s.anti_aesthetic, s.pixel_sludge)),
            "pixel_quality": s.pixel_quality,
            "pixel_sharpness": s.pixel_sharpness,
            "softness": s.texture_balance,
            "anti_crystal": _clip01(1.0 - s.crystal_affinity),
        }
        clip_blend = (
            0.22 * facets["clip_overall"]
            + 0.14 * facets["striking"]
            + 0.16 * facets["harmonious"]
            + 0.12 * facets["vivid"]
            + 0.12 * facets["composition"]
            + 0.10 * facets["softness"]
        )
        grounded_blend = (
            0.22 * facets["reality_resonance"]
            + 0.18 * facets["clarity"]
            + 0.15 * facets["structure"]
            + 0.10 * facets["observer"]
            + 0.10 * facets["anti_crystal"]
            + 0.25 * facets["pixel_quality"]
        )
        # Normalize blend weights
        total_w = clip_w + ground_w
        beauty = (clip_w * clip_blend + ground_w * grounded_blend) / total_w
        beauty *= (0.55 + 0.45 * facets["anti_sludge"])
        # Soft counterweight: pure crystal knives lose ~20% beauty
        beauty *= (0.80 + 0.20 * facets["softness"])
        self.last_facets = {k: float(v) for k, v in facets.items()}
        self.last_facets["clip_confidence"] = float(s.clip_confidence)
        return _clip01(beauty)


class NoveltyHead(RewardHead):
    """Novelty = new territory that is still legible, not pure noise."""
    def __init__(self):
        super().__init__("Novelty", base_weight=0.30)

    def evaluate(self, concept_embedding, latent, memory, signals: SharedSignals = None, **kwargs) -> float:
        s = signals
        geometric = _clip01(s.avg_nn_dist / 3.0)
        facets = {
            "geometric": geometric,
            "grounded": s.grounded_novelty,
            "depart_parents": _clip01(1.0 - s.parent_alignment),
            "under_explored": s.under_explored,
            "not_sludge": _clip01(1.0 - max(s.anti_aesthetic, s.pixel_sludge)),
            "tag_fresh": _clip01(1.0 - s.tag_peakiness * 0.5),  # less generic peak
            "soft_frontier": s.soft_affinity,  # soft regimes are under-explored vs crystal
            "not_all_crystal": _clip01(1.0 - 0.7 * s.crystal_affinity),
        }
        novelty = (
            0.25 * facets["geometric"]
            + 0.20 * facets["grounded"]
            + 0.12 * facets["depart_parents"]
            + 0.12 * facets["under_explored"]
            + 0.08 * facets["not_sludge"]
            + 0.05 * facets["tag_fresh"]
            + 0.10 * facets["soft_frontier"]
            + 0.08 * facets["not_all_crystal"]
        )
        self.last_facets = {k: float(v) for k, v in facets.items()}
        return _clip01(novelty)


class CoherenceHead(RewardHead):
    """Coherence = internal sense: density sweet-spot, lineage, tags, not chaos."""
    def __init__(self):
        super().__init__("Coherence", base_weight=0.25)

    def evaluate(self, concept_embedding, latent, memory, signals: SharedSignals = None, **kwargs) -> float:
        s = signals
        density_sweet = _clip01(1.0 - abs(s.avg_nn_dist - 1.0) / 2.0)
        facets = {
            "density_sweet": density_sweet,
            "lineage": s.parent_alignment,
            "semantic": s.tag_peakiness,
            "channel_structure": s.latent_channel_balance,
            "low_surprise_fail": _clip01(1.0 - min(1.0, s.prediction_error)),
            "not_chaos": _clip01(1.0 - s.anti_aesthetic * 0.7),
            "recombo_bridge": s.parent_spread * s.parent_alignment,  # fusion that still holds
        }
        coherence = (
            0.25 * facets["density_sweet"]
            + 0.20 * facets["lineage"]
            + 0.15 * facets["semantic"]
            + 0.15 * facets["channel_structure"]
            + 0.10 * facets["low_surprise_fail"]
            + 0.10 * facets["not_chaos"]
            + 0.05 * facets["recombo_bridge"]
        )
        self.last_facets = {k: float(v) for k, v in facets.items()}
        return _clip01(coherence)


class CompressionHead(RewardHead):
    """Compression = rich meaning in a tight code: entropy, tags, energy."""
    def __init__(self):
        super().__init__("Compression", base_weight=0.10)

    def evaluate(self, concept_embedding, latent, memory, signals: SharedSignals = None, **kwargs) -> float:
        s = signals
        facets = {
            "embedding_entropy": s.embedding_entropy,
            "tag_peak": s.tag_peakiness,
            "energy_efficiency": _clip01(s.embedding_norm * (1.0 - abs(s.latent_energy - 0.5))),
            "channel_balance": s.latent_channel_balance,
            "not_redundant": _clip01(s.avg_nn_dist / 2.5),  # not a near-duplicate
        }
        compression = (
            0.30 * facets["embedding_entropy"]
            + 0.25 * facets["tag_peak"]
            + 0.20 * facets["energy_efficiency"]
            + 0.15 * facets["channel_balance"]
            + 0.10 * facets["not_redundant"]
        )
        self.last_facets = {k: float(v) for k, v in facets.items()}
        return _clip01(compression)


class MemoryHead(RewardHead):
    """Memory = belonging to lineage / reality / attended story."""
    def __init__(self):
        super().__init__("Memory", base_weight=0.10)

    def evaluate(self, concept_embedding, latent, memory, signals: SharedSignals = None, **kwargs) -> float:
        s = signals
        facets = {
            "parent_bond": s.parent_alignment if s.parent_alignment > 0.25 else 0.2,
            "reality_root": s.reality_affinity,
            "local_familiarity": s.local_density,
            "observer_thread": s.observer_boost,
            "stable_recall": s.latent_channel_balance * s.tag_peakiness,
        }
        memory_score = (
            0.35 * facets["parent_bond"]
            + 0.25 * facets["reality_root"]
            + 0.15 * facets["local_familiarity"]
            + 0.15 * facets["observer_thread"]
            + 0.10 * facets["stable_recall"]
        )
        self.last_facets = {k: float(v) for k, v in facets.items()}
        return _clip01(memory_score)


class CuriosityHead(RewardHead):
    """Curiosity = useful surprise: PE, uncertainty, frontier near attention."""
    def __init__(self):
        super().__init__("Curiosity", base_weight=0.20)

    def evaluate(self, concept_embedding, latent, memory, signals: SharedSignals = None, **kwargs) -> float:
        s = signals
        facets = {
            "uncertainty": s.under_explored,
            "prediction_error": _clip01(s.prediction_error / 0.8),
            "frontier": s.grounded_novelty,
            "observer_pull": s.observer_boost,
            "not_pure_noise": _clip01(1.0 - s.anti_aesthetic),
            "learnable": _clip01(s.tag_peakiness * 0.5 + s.latent_channel_balance * 0.5),
        }
        curiosity = (
            0.25 * facets["uncertainty"]
            + 0.25 * facets["prediction_error"]
            + 0.20 * facets["frontier"]
            + 0.15 * facets["observer_pull"]
            + 0.10 * facets["not_pure_noise"]
            + 0.05 * facets["learnable"]
        )
        self.last_facets = {k: float(v) for k, v in facets.items()}
        return _clip01(curiosity)


def couple_head_rewards(raw: Dict[str, float], signals: SharedSignals) -> Dict[str, float]:
    """
    Sibling coupling: each head's final score is informed by related heads
    so Beauty isn't blind to Coherence, Novelty isn't pure noise, etc.
    """
    b = raw.get("Beauty", 0.5)
    n = raw.get("Novelty", 0.5)
    c = raw.get("Coherence", 0.5)
    k = raw.get("Curiosity", 0.5)
    m = raw.get("Memory", 0.5)
    p = raw.get("Compression", 0.5)

    coupled = {
        # Beautiful chaos is discounted; beautiful structure is amplified
        "Beauty": _clip01(0.70 * b + 0.15 * c + 0.10 * (1.0 - signals.anti_aesthetic) + 0.05 * p),
        # Novel sludge is discounted; novel-but-coherent is rewarded
        "Novelty": _clip01(0.65 * n + 0.15 * k + 0.10 * c + 0.10 * (1.0 - signals.anti_aesthetic)),
        # Coherence that ignores beauty/memory becomes sterile
        "Coherence": _clip01(0.70 * c + 0.15 * m + 0.10 * b + 0.05 * p),
        # Compression informed by whether meaning is actually there
        "Compression": _clip01(0.75 * p + 0.15 * c + 0.10 * b),
        # Memory enriched by beauty of what is remembered
        "Memory": _clip01(0.70 * m + 0.15 * c + 0.15 * b),
        # Curiosity steered toward learnable frontiers, not noise
        "Curiosity": _clip01(0.65 * k + 0.15 * n + 0.10 * c + 0.10 * (1.0 - signals.anti_aesthetic)),
    }
    return coupled


class MetaJudge:
    """
    Dynamic reward weighting system with anti-spam bias.
    Builds SharedSignals once, scores all heads, couples them, then weights.
    """
    def __init__(self, heads: List[RewardHead]):
        self.heads = heads
        self.head_dict = {h.name: h for h in heads}
        self.current_time = 0
        self.decision_history = []
        self.last_signals: Optional[SharedSignals] = None
        self.last_facets: Dict[str, Dict[str, float]] = {}

    def compute_dynamic_weights(self) -> Dict[str, float]:
        weights = {}
        for head in self.heads:
            w = head.base_weight
            time_since_activation = self.current_time - head.last_activation
            scarcity_boost = min(1.5, 1.0 + (time_since_activation / 100.0))
            recent_activations = sum(
                1 for t, r in head.history
                if self.current_time - t < 50 and r > 0.7
            )
            fatigue_penalty = max(0.3, 1.0 - (recent_activations / 20.0))
            if len(head.history) > 0:
                recent_rewards = [r for t, r in head.history[-20:]]
                avg_recent = np.mean(recent_rewards)
                urgency = max(0.5, 1.0 - avg_recent)
            else:
                urgency = 1.0
            weights[head.name] = w * scarcity_boost * fatigue_penalty * urgency

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            weights = {h.name: 1.0 / len(self.heads) for h in self.heads}
        return weights

    def evaluate_concept(
        self,
        concept_embedding: np.ndarray,
        latent: torch.Tensor,
        memory,
        custom_weights: Dict[str, float] = None,
        **kwargs,
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        self.current_time += 1

        if custom_weights is not None:
            weights = custom_weights
        else:
            weights = self.compute_dynamic_weights()

        signals = build_shared_signals(
            concept_embedding, latent, memory, **kwargs
        )
        self.last_signals = signals

        raw_rewards = {}
        facets = {}
        for head in self.heads:
            reward = head.evaluate(
                concept_embedding, latent, memory, signals=signals, **kwargs
            )
            raw_rewards[head.name] = reward
            facets[head.name] = dict(head.last_facets)

        head_rewards = couple_head_rewards(raw_rewards, signals)
        self.last_facets = facets

        for head in self.heads:
            head.record_reward(head_rewards[head.name], self.current_time)

        total_reward = sum(
            weights.get(name, 0) * head_rewards[name] for name in head_rewards.keys()
        )

        self.decision_history.append({
            "time": self.current_time,
            "rewards": head_rewards.copy(),
            "raw_rewards": raw_rewards.copy(),
            "weights": weights.copy(),
            "total": total_reward,
            "facets": facets,
        })
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]

        return total_reward, head_rewards, weights

    def get_status_summary(self) -> str:
        weights = self.compute_dynamic_weights()
        lines = ["Multi-Head RL Status:"]
        for head in sorted(self.heads, key=lambda h: weights[h.name], reverse=True):
            w = weights[head.name]
            recent = head.history[-1][1] if head.history else 0.0
            lines.append(
                f"  {head.name:12s}: weight={w:.3f}  recent={recent:.3f}  "
                f"acts={head.total_activations}"
            )
        if self.last_signals is not None:
            s = self.last_signals
            lines.append(
                f"  signals: aesthetic={s.aesthetic_overall:.2f} "
                f"anti={s.anti_aesthetic:.2f} pixel_q={s.pixel_quality:.2f} "
                f"pixel_sludge={s.pixel_sludge:.2f} reality={s.reality_distance:.2f} "
                f"observer={s.observer_boost:.2f}"
            )
        return "\n".join(lines)


def create_multihead_system() -> MetaJudge:
    heads = [
        BeautyHead(),
        NoveltyHead(),
        CoherenceHead(),
        CompressionHead(),
        MemoryHead(),
        CuriosityHead(),
    ]
    return MetaJudge(heads)


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from database import WorldMemory

    print("Testing Multi-Head RL System (multi-signal + coupling)")
    print("=" * 60)

    judge = create_multihead_system()
    memory = WorldMemory()

    test_embedding = np.random.randn(512).astype(np.float32)
    test_embedding = test_embedding / np.linalg.norm(test_embedding)
    test_latent = torch.randn(1, 4, 64, 64)

    total, rewards, weights = judge.evaluate_concept(
        test_embedding,
        test_latent,
        memory,
        aesthetic_scores={
            "overall": 0.7,
            "striking": 0.6,
            "harmonious": 0.65,
            "vivid": 0.55,
            "composition": 0.6,
            "anti": 0.1,
        },
        reality_distance=1.0,
        prediction_error=0.2,
        observer_boost=0.3,
    )

    print(f"\nTotal Reward: {total:.3f}")
    print("\nCoupled Head Rewards:")
    for name, reward in rewards.items():
        print(f"  {name:12s}: {reward:.3f}")
    print("\n" + judge.get_status_summary())
    print("\n[OK] Multi-signal multi-head RL operational")
