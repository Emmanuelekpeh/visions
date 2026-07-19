"""
Meta-Controller
The missing seventh module that coordinates the six RL heads.

The question: "Today we need novelty more than coherence" - who decides?

Inputs:
- Novelty trend (are we stagnating?)
- Memory saturation (is it full?)
- Concept age (young vs mature ecosystem)
- Replay frequency (are we stuck?)
- Reconstruction loss (are concepts decodable?)
- Entropy (diversity vs convergence)

Outputs:
- Dynamic weights for each RL head
- Exploration vs exploitation mode
- Validation strictness
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class SystemState:
    """Current state of the universe for meta-control."""
    novelty_trend: float          # -1 to 1: declining vs rising novelty
    memory_saturation: float      # 0 to 1: how full is memory
    avg_concept_age: float        # Average generation age
    replay_frequency: float       # 0 to 1: how often repeating concepts
    reconstruction_loss: float    # 0 to 1: decode quality
    embedding_entropy: float      # 0 to 1: diversity measure
    prediction_error_trend: float # -1 to 1: learning rate
    boredom: float = 0.0          # 0 to 1: Phase 6 True Boredom metric


class MetaController:
    """
    Adaptive controller that sets RL head weights based on system state.
    This is the "dopamine system" - it decides what matters right now.
    """
    
    def __init__(self):
        # Base weights (defaults)
        self.base_weights = {
            "Beauty": 0.20,
            "Novelty": 0.25,
            "Coherence": 0.20,
            "Compression": 0.10,
            "Memory": 0.10,
            "Curiosity": 0.15
        }
        
        # History tracking
        self.novelty_history = []
        self.entropy_history = []
        self.prediction_error_history = []
        
        # Adaptation parameters
        self.adaptation_rate = 0.1  # How fast to adapt weights
        
        # Phase 5: Observer Attention
        self.attended_concept = None
        self.attention_energy = 0.0
        
    def focus_attention(self, concept_id: int):
        """Inject observer attention into a specific concept."""
        self.attended_concept = concept_id
        self.attention_energy = 1.0 # Max energy
        
    def compute_system_state(self, memory_system, rl_system, 
                             recent_concepts: List) -> SystemState:
        """
        Analyze current universe state to inform weight adaptation.
        """
        # 1. Novelty trend
        if len(self.novelty_history) > 10:
            recent_novelty = np.mean(self.novelty_history[-10:])
            older_novelty = np.mean(self.novelty_history[-20:-10]) if len(self.novelty_history) > 20 else recent_novelty
            novelty_trend = (recent_novelty - older_novelty) / (older_novelty + 1e-6)
        else:
            novelty_trend = 0.0
            
        # 2. Memory saturation
        total_concepts = len(memory_system.neurons) if hasattr(memory_system, 'neurons') else 1000
        max_concepts = getattr(memory_system, 'max_concepts', 5000)
        memory_saturation = total_concepts / max_concepts
        
        # 3. Average concept age
        if hasattr(memory_system, 'neurons'):
            ages = [n.age for n in memory_system.neurons.values()]
            avg_age = np.mean(ages) if ages else 0.0
        else:
            avg_age = 0.0
            
        # 4. Replay frequency (how often selecting same concepts)
        # Would track concept selection history in production
        replay_frequency = 0.3  # Placeholder
        
        # 5. Reconstruction loss (placeholder - would track decode quality)
        reconstruction_loss = 0.2  # Placeholder
        
        # 6. Embedding entropy
        if len(self.entropy_history) > 0:
            embedding_entropy = self.entropy_history[-1]
        else:
            embedding_entropy = 0.7  # Placeholder
            
        # 7. Prediction error trend
        if len(self.prediction_error_history) > 10:
            recent_pe = np.mean(self.prediction_error_history[-10:])
            older_pe = np.mean(self.prediction_error_history[-20:-10]) if len(self.prediction_error_history) > 20 else recent_pe
            pe_trend = (recent_pe - older_pe) / (older_pe + 1e-6)
        else:
            pe_trend = 0.0
            
        # Phase 6: True Boredom
        # Boredom = (1.0 - Prediction_Error) * (1.0 - Surprise) * Replay_Frequency
        recent_pe = np.mean(self.prediction_error_history[-10:]) if self.prediction_error_history else 0.5
        prediction_accuracy = max(0.0, 1.0 - recent_pe)
        low_surprise = max(0.0, 1.0 - abs(pe_trend))
        boredom = prediction_accuracy * low_surprise * replay_frequency
            
        return SystemState(
            novelty_trend=np.clip(novelty_trend, -1, 1),
            memory_saturation=memory_saturation,
            avg_concept_age=avg_age,
            replay_frequency=replay_frequency,
            reconstruction_loss=reconstruction_loss,
            embedding_entropy=embedding_entropy,
            prediction_error_trend=np.clip(pe_trend, -1, 1),
            boredom=boredom
        )
        
    def adapt_weights(self, state: SystemState) -> Dict[str, float]:
        """
        Core adaptive logic: Decide what matters RIGHT NOW.
        
        This is the missing piece - the coordinator that prevents
        six disconnected dictators.
        """
        weights = self.base_weights.copy()
        
        # Phase 5: Observer Attention
        if hasattr(self, 'attention_energy') and self.attention_energy > 0.1:
            print(f"[MetaController] Observer Attention Active ({self.attention_energy:.2f}) - tilting telescope!")
            # The observer is curious. Don't just clone the image (which would be Coherence/Memory).
            # Force the universe to explore the physics around this concept.
            weights["Curiosity"] += 0.30 * self.attention_energy
            weights["Novelty"] += 0.20 * self.attention_energy
            weights["Coherence"] -= 0.15 * self.attention_energy
            
            # Decay attention energy over time
            self.attention_energy *= 0.85
        
        # RULE 1: Stagnation → Increase Novelty (LESS SENSITIVE)
        if state.novelty_trend < -0.4:  # Novelty declining significantly (increased from -0.2)
            print("[MetaController] Detected stagnation - boosting Novelty")
            weights["Novelty"] += 0.15
            weights["Curiosity"] += 0.10
            weights["Coherence"] -= 0.15  # Relax coherence to allow exploration
            
        # RULE 2: Chaos → Increase Coherence
        if state.embedding_entropy > 0.8 and state.reconstruction_loss > 0.4:
            print("[MetaController] Detected chaos - boosting Coherence")
            weights["Coherence"] += 0.20
            weights["Beauty"] += 0.10
            weights["Novelty"] -= 0.15  # Reduce exploration
            
        # RULE 3: Memory Full → Increase Compression & Quality
        if state.memory_saturation > 0.85:
            print("[MetaController] Memory saturated - boosting Compression & Memory")
            weights["Compression"] += 0.15
            weights["Memory"] += 0.10
            weights["Curiosity"] -= 0.10  # Reduce new exploration
            
        # RULE 4: Mature Ecosystem → Balance All Objectives
        if state.avg_concept_age > 5000:
            print("[MetaController] Mature ecosystem - balancing objectives")
            # Move toward uniform weights
            for key in weights:
                weights[key] = 0.8 * weights[key] + 0.2 * (1.0 / len(weights))
                
        # RULE 5: High Prediction Error → Boost Curiosity (Learning Phase)
        if state.prediction_error_trend > 0.3:  # Lots of surprises = learning
            print("[MetaController] High learning signal - boosting Curiosity")
            weights["Curiosity"] += 0.15
            weights["Novelty"] += 0.10
            
        # RULE 6: True Boredom (Phase 6)
        if hasattr(state, 'boredom') and state.boredom > 0.8:
            if not hasattr(self, '_boredom_print_counter'):
                self._boredom_print_counter = 0
            self._boredom_print_counter += 1
            if self._boredom_print_counter % 10 == 0:
                print(f"[MetaController] TRUE BOREDOM ({state.boredom:.2f}) - violently shaking awake!")
            # Massive spike in exploration
            weights["Novelty"] += 0.30
            weights["Curiosity"] += 0.20
            weights["Coherence"] -= 0.20
            weights["Beauty"] -= 0.10
            
        # RULE 7: Replay Frequency High → Too Repetitive
        if state.replay_frequency > 0.6:
            print("[MetaController] High replay - forcing diversity")
            weights["Novelty"] += 0.15
            weights["Curiosity"] += 0.10
            
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        return weights
        
    def get_exploration_mode(self, state: SystemState) -> str:
        """
        Decide exploration vs exploitation strategy.
        """
        # Young ecosystem: Explore
        if state.avg_concept_age < 1000:
            return "EXPLORE"
            
        # Stagnating: Explore (LESS SENSITIVE)
        if state.novelty_trend < -0.5:  # Increased from -0.3
            return "EXPLORE"
            
        # Chaotic: Exploit (consolidate)
        if state.reconstruction_loss > 0.5:
            return "EXPLOIT"
            
        # Full memory: Exploit (refine)
        if state.memory_saturation > 0.9:
            return "EXPLOIT"
            
        # Default: Balanced
        return "BALANCED"
        
    def get_validation_strictness(self, state: SystemState) -> float:
        """
        How strict should acceptance be?
        Returns threshold ∈ [0.3, 0.7]
        """
        base_threshold = 0.4
        
        # Full memory → Be picky
        if state.memory_saturation > 0.85:
            base_threshold += 0.15
            
        # Stagnating → Be lenient (LESS SENSITIVE)
        if state.novelty_trend < -0.4:  # Increased from -0.2
            base_threshold -= 0.10
            
        # Chaotic → Be strict
        if state.reconstruction_loss > 0.4:
            base_threshold += 0.10
            
        return np.clip(base_threshold, 0.3, 0.7)
        
    def record_observation(self, novelty: float, entropy: float, 
                          prediction_error: float):
        """Track observations for trend analysis."""
        self.novelty_history.append(novelty)
        self.entropy_history.append(entropy)
        self.prediction_error_history.append(prediction_error)
        
        # Keep last 100
        if len(self.novelty_history) > 100:
            self.novelty_history = self.novelty_history[-100:]
        if len(self.entropy_history) > 100:
            self.entropy_history = self.entropy_history[-100:]
        if len(self.prediction_error_history) > 100:
            self.prediction_error_history = self.prediction_error_history[-100:]
            
    def get_status_summary(self, state: SystemState, weights: Dict[str, float]) -> str:
        """Generate human-readable status."""
        lines = ["[META-CONTROLLER STATUS]"]
        lines.append(f"  Novelty Trend: {state.novelty_trend:+.3f}")
        lines.append(f"  Memory: {state.memory_saturation*100:.1f}% full")
        lines.append(f"  Entropy: {state.embedding_entropy:.3f}")
        lines.append(f"  Prediction Error Trend: {state.prediction_error_trend:+.3f}")
        lines.append(f"\n  Active Weights:")
        for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"    {name:12s}: {weight:.3f}")
        return "\n".join(lines)


# Test
if __name__ == "__main__":
    print("Meta-Controller: The Missing Seventh Module")
    print("=" * 60)
    
    # Simulate different states
    controller = MetaController()
    
    # Scenario 1: Stagnation
    print("\nScenario 1: STAGNATION (novelty declining)")
    state1 = SystemState(
        novelty_trend=-0.5,
        memory_saturation=0.6,
        avg_concept_age=2000,
        replay_frequency=0.7,
        reconstruction_loss=0.2,
        embedding_entropy=0.4,
        prediction_error_trend=0.0
    )
    weights1 = controller.adapt_weights(state1)
    print(f"  Novelty weight: {weights1['Novelty']:.3f} (boosted)")
    print(f"  Coherence weight: {weights1['Coherence']:.3f} (reduced)")
    
    # Scenario 2: Chaos
    print("\nScenario 2: CHAOS (high entropy, poor reconstruction)")
    state2 = SystemState(
        novelty_trend=0.3,
        memory_saturation=0.5,
        avg_concept_age=1500,
        replay_frequency=0.3,
        reconstruction_loss=0.6,
        embedding_entropy=0.9,
        prediction_error_trend=0.2
    )
    weights2 = controller.adapt_weights(state2)
    print(f"  Coherence weight: {weights2['Coherence']:.3f} (boosted)")
    print(f"  Novelty weight: {weights2['Novelty']:.3f} (reduced)")
    
    # Scenario 3: Memory Full
    print("\nScenario 3: MEMORY FULL (need compression)")
    state3 = SystemState(
        novelty_trend=0.1,
        memory_saturation=0.92,
        avg_concept_age=4000,
        replay_frequency=0.5,
        reconstruction_loss=0.3,
        embedding_entropy=0.6,
        prediction_error_trend=-0.1
    )
    weights3 = controller.adapt_weights(state3)
    print(f"  Compression weight: {weights3['Compression']:.3f} (boosted)")
    print(f"  Memory weight: {weights3['Memory']:.3f} (boosted)")
    
    print("\n[Meta-Controller operational - coordinates 6 RL heads dynamically]")
