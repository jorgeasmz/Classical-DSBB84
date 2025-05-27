import numpy as np

class Attacker:
    """
    Simulates attacks on QKD protocols
    """
    
    @staticmethod
    def pns_attack(photon_nums: np.ndarray) -> np.ndarray:
        """
        Perform a Photon Number Splitting (PNS) attack
        
        Args:
            photon_nums: Original number of photons sequence
            
        Returns:
            Modified number of photons after attack
        """
        modified_photons = photon_nums.copy()
        
        # For each position with more than 1 photon, keep only 1
        # For positions with 0 or 1 photon, set to 0
        for i in range(len(photon_nums)):
            if photon_nums[i] <= 1:
                modified_photons[i] = 0  # Keep no photons for single-photon or vacuum states
            else:
                modified_photons[i] = photon_nums[i] - 1  # Keep all but one photon
        
        return modified_photons