import numpy as np
from typing import List, Tuple, Optional

class Source:
    """
    Simulates a quantum photon source for QKD protocols
    Capable of generating regular and decoy photon states with reproducible sequences
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize a photon source with an optional seed
        
        Args:
            seed: Random seed for reproducible sequences. If None, sequences will be truly random.
        """
        self.rng = np.random.RandomState(seed)
    
    def generate_decoy_photons(self, 
                               mu: float, 
                               nu: List[float], 
                               length: int, 
                               decoy_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate complete photon sequences with decoy states for BB84 protocol using the initialized seed
        
        Args:
            mu: Mean photon number for signal state
            nu: List of mean photon numbers for decoy states
            length: Length of the sequence to generate
            decoy_ratio: Ratio of decoy states
            
        Returns:
            Tuple of (bit sequence, base choices, photon numbers, decoy sequence)
        """
        decoy_state_num = len(nu)
        
        # Ensure we have at least 2 decoy states
        if decoy_state_num < 2:
            nu = list(nu) + [0]
            decoy_state_num += 1
        
        # Prepare all intensity levels (signal + decoy states)
        decoy_nu = [mu] + list(nu)
        
        # Generate complete decoy sequence
        probs = [1 - decoy_ratio] + [decoy_ratio / decoy_state_num] * decoy_state_num
        states = list(range(1, decoy_state_num + 2))  # 1-indexed states
        decoy_seq = self.rng.choice(states, size=length, p=probs)
        
        # Generate bit and base sequences (these are independent of the decoy states)
        bit_seq = self.rng.randint(0, 2, size=length)
        base_choices = self.rng.randint(0, 2, size=length)
        
        # Generate photon numbers based on the intensity for each position
        photon_nums = np.zeros(length, dtype=int)
        
        for i in range(length):
            intensity = decoy_nu[decoy_seq[i] - 1]  # Adjust for 1-indexing
            photon_nums[i] = self.rng.poisson(intensity)
            
        return bit_seq, base_choices, photon_nums, decoy_seq