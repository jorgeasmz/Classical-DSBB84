import numpy as np
import json
import os

from simulator import BB84Simulator
from tqdm import tqdm

def eta_state(mpn: float, state_gain: float, dark_count_rate: float) -> float:
    """
    Calculate the state efficiency for a specific state type ('signal', 'decoy', or 'vacuum').

    Args:
        mpn (float): Mean photon number.
        state_gain (float): The gain of the state.
        dark_count_rate (float): The dark count rate.

    Returns:
        float: The calculated efficiency for the specified state type.
    """
    return -np.log(np.abs(1 + dark_count_rate - state_gain)) / mpn

def yield_i(n: int, eta_state: float, dark_count_rate: float) -> float:
    """
    Calculate the yield of n-photon states.

    Args:
        n (int): The number of photons in the state.
        eta_state (float): The efficiency of the state.
        dark_count_rate (float): The dark count rate.

    Returns:
        float: The calculated yield of n-photon states.
    """
    eta_n = (1 - (1 - eta_state)**n)
    return dark_count_rate + eta_n - (dark_count_rate * eta_n)

def generate_filename(num_runs, sequence_length, attack_type):
    """
    Generate a filename with the specified format.
    
    Args:
        num_runs: Number of simulation runs
        sequence_length: Length of photon sequence
        attack_type: Type of attack used ("PNS", "BS", or "No-Eve")
        
    Returns:
        String with the generated filename
    """
    return f"Runs_{num_runs}_Length_{sequence_length}_{attack_type}.json"

def save_yields_to_file(y_expected, y_signal, y_decoy, n_values, 
                        num_runs, sequence_length, mu, nu_1, nu_2, decoy_ratio,
                        alpha, l, channel_loss, receiver_loss, detector_efficiency,
                        dark_count_rate, detector_error, attack_type, 
                        eta, signal_gains, decoy_gains, signal_QBERs, decoy_QBERs):
    """
    Save yield data to a JSON file with specified format.
    
    Args:
        y_expected: List of lists with expected yields
        y_signal: List of lists with signal yields
        y_decoy: List of lists with decoy yields
        n_values: n values used
        num_runs: Number of simulation runs
        sequence_length: Length of photon sequence
        [Additional simulation parameters]
        attack_type: "PNS", "BS", or "No-Eve"
    """
    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory: {results_dir}")
    
    data = {
        'simulation_params': {
            'num_runs': num_runs,
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
            'attack_type': attack_type,
            'eta': eta
        },
        'signal_gains': signal_gains,
        'decoy_gains': decoy_gains,
        'signal_QBERs': signal_QBERs,
        'decoy_QBERs': decoy_QBERs,
        'n_values': n_values,
        'y_expected': [list(map(float, yields)) for yields in y_expected],
        'y_signal': [list(map(float, yields)) for yields in y_signal],
        'y_decoy': [list(map(float, yields)) for yields in y_decoy]
    }
    
    filename = generate_filename(num_runs, sequence_length, attack_type)
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Data saved to '{filepath}'")
    return filepath

def main():
    """Main function to run the BB84 QKD simulation"""
    
    # Fixed simulation parameters
    num_runs = 5000
    sequence_length = 25000
    mu = 5.722507
    nu_1 = 2.268934
    nu_2 = 0.0
    decoy_ratio = 0.25
    alpha = 0.0
    l = 20.0
    channel_loss = 0
    receiver_loss = 1.038
    detector_efficiency = 0.47 * 2
    dark_count_rate = 0.032765
    detector_error = 0.014
    
    # Eavesdropper settings - Choose one: "No-Eve", "PNS", or "BS"
    attack_type = "No-Eve"  # Change this to "BS" or "No-Eve" as needed
    eavesdropper = attack_type != "No-Eve"

    # Values of n for the yield calculation
    n_values = list(range(1, 6))

    # Initialize lists for QBERs and gains
    Q_mus = []
    Q_nu_1s = []
    E_mus = []
    E_nu_1s = []

    # Yield lists for expected, signal, and decoy states
    y_expected = [[] for _ in n_values]
    y_signal = [[] for _ in n_values]
    y_decoy = [[] for _ in n_values]

    for run in tqdm(range(num_runs), desc="Simulations", unit="sim"):
        # No seed is set for randomness
        seed = None

        # Initialize the BB84 simulator
        simulator = BB84Simulator(seed=seed)
        
        # Run the simulation
        results = simulator.run_simulation(
            sequence_length=sequence_length,  # Length of the photon sequence
            mu=mu,                            # Signal state intensity
            nu_1=nu_1,                        # First decoy state intensity
            nu_2=nu_2,                        # Second decoy state intensity
            decoy_ratio=decoy_ratio,          # Ratio of decoy states
            alpha=alpha,                      # Fiber attenuation coefficient (dB/km)
            l=l,                              # Length of the quantum channel (km)
            channel_loss=channel_loss,        # Loss in the channel (dB)
            receiver_loss=receiver_loss,      # Loss in the receiver's side (dB)
            detector_efficiency=detector_efficiency,  # Efficiency of the detector
            dark_count_rate=dark_count_rate,  # Dark count rate (Y0)
            detector_error=detector_error,    # Detector error probability (e_d)
            eavesdropper=eavesdropper,        # Whether to include an eavesdropper
            attack_type=attack_type.lower() if eavesdropper else "none",  # Type of attack (pns or bs)
        )
        
        eta = results['eta']
        dark_count_rate = results['dark_count_rate']

        # Calculate the state efficiency for signal and decoy states
        eta_signal = eta_state(results['mu'], results['Q_mu'], dark_count_rate)
        eta_decoy = eta_state(results['nu_1'], results['Q_nu_1'], dark_count_rate)

        # Store QBERs and gains
        Q_mus.append(results['Q_mu'])
        Q_nu_1s.append(results['Q_nu_1'])
        E_mus.append(results['E_mu'])
        E_nu_1s.append(results['E_nu_1'])
        
        # Calculate yields for each n value
        for i, n in enumerate(n_values):
            y_expected[i].append(yield_i(n, eta, dark_count_rate))
            y_signal[i].append(yield_i(n, eta_signal, dark_count_rate))
            y_decoy[i].append(yield_i(n, eta_decoy, dark_count_rate))
    
    # Save data to a file
    filepath = save_yields_to_file(
        y_expected, y_signal, y_decoy, n_values,
        num_runs, sequence_length, mu, nu_1, nu_2, decoy_ratio,
        alpha, l, channel_loss, receiver_loss, 
        detector_efficiency, dark_count_rate, detector_error,
        attack_type, eta, Q_mus, Q_nu_1s, E_mus, E_nu_1s
    )
    
    print(f"Simulation completed. Run 'python plot_results.py --file {os.path.basename(filepath)}' to generate the plot.")

if __name__ == "__main__":
    main()