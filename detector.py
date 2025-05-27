import numpy as np
from typing import Optional

class Detector:
    """
    Simulates a single photon detector for BB84 protocol
    """
    
    def __init__(self, 
                 transmittance: float, 
                 dark_count_rate: float, 
                 error_prob: float, 
                 seed: Optional[int] = None):
        """
        Initialize a single photon detector
        
        Args:
            transmittance: Overall transmission and detection efficiency (eta)
            dark_count_rate: Dark count rate (Y0)
            error_prob: Detector error probability
            seed: Random seed for reproducible detection results. If None, results will be truly random.
        """
        self.transmittance = transmittance
        self.dark_count_rate = dark_count_rate
        self.error_prob = error_prob
        self.rng = np.random.RandomState(seed)
    
    def detect(self, photon_seq: np.ndarray) -> np.ndarray:
        """
        Simulate photon detection for BB84 protocol
        
        Args:
            photon_seq: Array with shape (3, L) containing [bit_seq, basis_seq, number_seq]
            
        Returns:
            Result sequence where:
                -1: No detection
                -2: Wrong basis
                0/1: Detected bit value
        """
        bit_seq = photon_seq[0]
        basis_seq = photon_seq[1]
        number_seq = photon_seq[2]
        
        length = len(bit_seq)
        result_seq = np.zeros(length, dtype=int)
        
        # Generate measurement basiss with the same random generator
        measure_basis_seq = self.rng.randint(0, 2, size=length)
        
        # Process each photon
        for i in range(length):
            # Check if basiss match
            if measure_basis_seq[i] != basis_seq[i]:
                # Wrong basis
                result_seq[i] = -2
            elif number_seq[i] > 0:
                # There are photons to detect
                # Generate detection outcomes for each photon
                detection_probs = [1 - self.transmittance,                    #  0: Not detected
                                  self.transmittance * (1 - self.error_prob), #  1: Correctly detected
                                  self.transmittance * self.error_prob]       # -1: Incorrectly detected
                
                detected = self.rng.choice([0, 1, -1], 
                                          size=number_seq[i], 
                                          p=detection_probs)
                
                if -1 in detected:
                    # At least one photon incorrectly detected
                    result_seq[i] = (bit_seq[i] + 1) % 2

                elif 1 not in detected:
                    # No photons detected
                    # Consider dark count
                    result_seq[i] = self.rng.choice([bit_seq[i], (bit_seq[i] + 1) % 2, -1],
                                                    p=[self.dark_count_rate * 0.5, 
                                                       self.dark_count_rate * 0.5, 
                                                       1 - self.dark_count_rate])
                    
                else:
                    # At least one photon correctly detected
                    # Consider dark count
                    result_seq[i] = self.rng.choice([(bit_seq[i] + 1) % 2, bit_seq[i]], 
                                                    p=[self.dark_count_rate * 0.5, 
                                                       1 - self.dark_count_rate * 0.5])
            else:
                # No photons
                # Consider dark count
                result_seq[i] = self.rng.choice([bit_seq[i], (bit_seq[i] + 1) % 2, -1], 
                                               p=[self.dark_count_rate * 0.5, 
                                                  self.dark_count_rate * 0.5, 
                                                  1 - self.dark_count_rate])
        
        return result_seq