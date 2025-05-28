import numpy as np

from typing import Dict, Optional
from source import Source
from detector import Detector
from attacker import Attacker

class BB84Simulator:
    """
    Main simulator for BB84 QKD protocol with decoy states
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the BB84 simulator with optional seed for reproducibility
        
        Args:
            seed: Random seed for reproducible simulations
        """
        self.seed = seed
    
    def run_simulation(self, 
                       sequence_length: 100000,
                       eavesdropper: bool = False,
                       attack_type: str = 'pns',
                       mu: float = 0.65, 
                       nu_1: float = 0.08, 
                       nu_2: float = 1e-5,
                       decoy_ratio: float = 0.25,
                       alpha: float = 0.2,
                       l: float = 20.0,
                       channel_loss: float = 5.6,
                       receiver_loss: float = 3.5,
                       detector_efficiency: float = 0.10,
                       dark_count_rate: float = 1e-5,
                       detector_error: float = 0.014) -> Dict:
        """
        Run a BB84 QKD simulation with decoy states over different distances
        
        Args:
            sequence_length: Length of the photon sequence
            mu: Signal state intensity
            nu_1: First decoy state intensity
            nu_2: Second decoy state intensity
            decoy_ratio: Ratio of decoy states
            alpha: Fiber attenuation coefficient (dB/km)
            l: Length of the quantum channel (km)
            receiver_loss: Loss in the receiver's side (dB)
            detector_efficiency: Efficiency of the detector
            dark_count_rate: Dark count rate (Y0)
            detector_error: Detector error probability (e_d)
            eavesdropper: Whether to include an eavesdropper
            
        Returns:
            Dictionary containing simulation results
        """

        # Quantum channel transmittance
        if channel_loss:
            t_AB = 10**((-channel_loss)/10)
        else:
            t_AB = 10**((-alpha * l)/10)
  
        t_B = 10**((-receiver_loss)/10) # Optical elements transmittance
        eta_D = detector_efficiency # Detector efficiency
        eta_Bob = t_B * eta_D # Bob's side transmittance
        eta = t_AB * eta_Bob # * (1 - decoy_ratio) # Overall transmittance
        fraction = 0.9 # Fraction of transmitted pulse energy due to a BS attack

        # Initialize the source
        photon_source = Source(seed=self.seed)
        bit_seq, base_choices, photon_nums, decoy_seq = photon_source.generate_decoy_photons(mu, [nu_1, nu_2], sequence_length, decoy_ratio=decoy_ratio)

        # If eavesdropper is present, simulate the attack
        if eavesdropper:
            original_eta = eta
            if attack_type == 'pns':
                attacker = Attacker()
                photon_nums = attacker.pns_attack(photon_nums)
            else:
                eta = eta * fraction
        
        # Simulate detection
        detector = Detector(transmittance=eta, 
                            dark_count_rate=dark_count_rate, 
                            error_prob=detector_error, 
                            seed=self.seed)
        detection_results = detector.detect(np.vstack([bit_seq, base_choices, photon_nums]))

        # Analyze results
        match_seq = detection_results == bit_seq
        rate = np.sum(match_seq == 1) / len(bit_seq)
        
        # Select only valid measurements (matching bases)
        valid_index = detection_results != -2
        valid_seq = detection_results[valid_index]
        valid_src_seq = bit_seq[valid_index]
        valid_decoy_seq = decoy_seq[valid_index]
        
        # Separate results by state type
        signal_indices = valid_decoy_seq == 1
        decoy_1_indices = valid_decoy_seq == 2
        decoy_2_indices = valid_decoy_seq == 3
        
        signal_seq = valid_seq[signal_indices]
        decoy_1_seq = valid_seq[decoy_1_indices]
        decoy_2_seq = valid_seq[decoy_2_indices]
        
        signal_src_seq = valid_src_seq[signal_indices]
        decoy_1_src_seq = valid_src_seq[decoy_1_indices]
        decoy_2_src_seq = valid_src_seq[decoy_2_indices]

        # Find detected signals
        detected_signal_index = signal_seq != -1
        detected_decoy_1_index = decoy_1_seq != -1
        detected_decoy_2_index = decoy_2_seq != -1
        
        detected_signal_seq = signal_seq[detected_signal_index]
        detected_decoy_1_seq = decoy_1_seq[detected_decoy_1_index]
        detected_decoy_2_seq = decoy_2_seq[detected_decoy_2_index]
        
        detected_signal_src_seq = signal_src_seq[detected_signal_index]
        detected_decoy_1_src_seq = decoy_1_src_seq[detected_decoy_1_index]
        detected_decoy_2_src_seq = decoy_2_src_seq[detected_decoy_2_index]

        Q_mu = np.sum(detected_signal_index) / len(signal_seq) if len(signal_seq) > 0 else 0
        Q_nu_1 = np.sum(detected_decoy_1_index) / len(decoy_1_seq) if len(decoy_1_seq) > 0 else 0
        Q_nu_2 = np.sum(detected_decoy_2_index) / len(decoy_2_seq) if len(decoy_2_seq) > 0 else 0

        # Calculate error rates
        if len(detected_signal_src_seq) > 0:
            E_mu = np.sum(detected_signal_seq != detected_signal_src_seq) / len(detected_signal_src_seq)
        else:
            E_mu = 0
            
        if len(detected_decoy_1_src_seq) > 0:
            E_nu1 = np.sum(detected_decoy_1_seq != detected_decoy_1_src_seq) / len(detected_decoy_1_src_seq)
        else:
            E_nu1 = 0
            
        if len(detected_decoy_2_src_seq) > 0:
            E_nu2 = np.sum(detected_decoy_2_seq != detected_decoy_2_src_seq) / len(detected_decoy_2_src_seq)
        else:
            E_nu2 = 0

        simulation_results = {
            'sequence_length': sequence_length,
            'mu': mu,
            'nu_1': nu_1,
            'nu_2': nu_2,
            'decoy_ratio': decoy_ratio,
            'alpha': alpha,
            'l': l,
            'channel_loss': channel_loss,
            'receiver_loss': receiver_loss,
            'detector_efficiency': detector_efficiency,
            'dark_count_rate': dark_count_rate,
            'detector_error': detector_error,
            'eavesdropper': eavesdropper,
            'eta': original_eta if eavesdropper else eta,
            'Q_mu': Q_mu,
            'Q_nu_1': Q_nu_1,
            'Q_nu_2': Q_nu_2,
            'E_mu': E_mu,
            'E_nu1': E_nu1,
            'E_nu2': E_nu2
        }

        return simulation_results