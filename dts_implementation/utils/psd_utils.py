"""
Power Spectral Density (PSD) Utilities for S-ADT

Implements spectral analysis for time series to compute the spectral reward.

Mathematical Framework from S-ADT.md:
- S(ω) = |F(x)(ω)|² : Power Spectral Density
- Spectral penalty: ∫ |log S_x₀(ω) - log E[S_c(ω)]| dω
"""

import torch
import numpy as np
from typing import Union, Optional, Tuple, Dict
from scipy import signal
from scipy.stats import wasserstein_distance


def compute_psd(
    time_series: Union[torch.Tensor, np.ndarray],
    sampling_rate: float = 1.0,
    nperseg: Optional[int] = None,
    method: str = 'welch'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density of time series
    
    Args:
        time_series: Time series data [batch, length] or [length]
        sampling_rate: Sampling rate in Hz
        nperseg: Length of each segment for Welch method
        method: 'welch' or 'periodogram'
    
    Returns:
        freqs: Frequency bins [n_freqs]
        psd: Power spectral density [batch, n_freqs] or [n_freqs]
    """
    # Convert to numpy if torch tensor
    if isinstance(time_series, torch.Tensor):
        time_series = time_series.detach().cpu().numpy()
    
    # Handle batching
    if time_series.ndim == 1:
        time_series = time_series[np.newaxis, :]
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, length = time_series.shape
    
    # Set default segment length
    if nperseg is None:
        nperseg = min(256, length // 4)
    
    # Compute PSD for each sample in batch
    psds = []
    freqs = None
    
    for i in range(batch_size):
        if method == 'welch':
            f, psd = signal.welch(
                time_series[i],
                fs=sampling_rate,
                nperseg=nperseg,
                scaling='density'
            )
        elif method == 'periodogram':
            f, psd = signal.periodogram(
                time_series[i],
                fs=sampling_rate,
                scaling='density'
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        psds.append(psd)
        if freqs is None:
            freqs = f
    
    psds = np.stack(psds, axis=0)
    
    if squeeze_output:
        psds = psds.squeeze(0)
    
    return freqs, psds


def compute_expected_psd(
    context_time_series: Union[torch.Tensor, np.ndarray],
    sampling_rate: float = 1.0,
    nperseg: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute expected PSD from historical context
    
    E[S_c(ω)] in the S-ADT formula.
    
    Args:
        context_time_series: Historical time series [batch, length] or [length]
        sampling_rate: Sampling rate
        nperseg: Segment length
    
    Returns:
        freqs: Frequency bins
        expected_psd: Mean PSD across batch
    """
    freqs, psds = compute_psd(context_time_series, sampling_rate, nperseg)
    
    # If batched, take mean across batch
    if psds.ndim == 2:
        expected_psd = np.mean(psds, axis=0)
    else:
        expected_psd = psds
    
    return freqs, expected_psd


def spectral_distance(
    psd1: np.ndarray,
    psd2: np.ndarray,
    freqs: np.ndarray,
    metric: str = 'wasserstein'
) -> float:
    """
    Compute distance between two PSDs
    
    Implements: ∫ |log S_x(ω) - log S_c(ω)| dω
    
    Args:
        psd1: First PSD [n_freqs]
        psd2: Second PSD [n_freqs]
        freqs: Frequency bins [n_freqs]
        metric: Distance metric ('wasserstein', 'l1', 'kl')
    
    Returns:
        distance: Spectral distance (lower = more similar)
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    psd1 = np.maximum(psd1, eps)
    psd2 = np.maximum(psd2, eps)
    
    if metric == 'wasserstein':
        # Wasserstein distance in log-frequency space
        log_psd1 = np.log(psd1)
        log_psd2 = np.log(psd2)
        
        # Normalize to probability distributions
        p1 = (log_psd1 - log_psd1.min()) + eps
        p1 = p1 / p1.sum()
        p2 = (log_psd2 - log_psd2.min()) + eps
        p2 = p2 / p2.sum()
        
        distance = wasserstein_distance(freqs, freqs, p1, p2)
    
    elif metric == 'l1':
        # L1 distance in log space (from S-ADT paper)
        log_diff = np.abs(np.log(psd1) - np.log(psd2))
        distance = np.trapz(log_diff, freqs)
    
    elif metric == 'kl':
        # KL divergence
        # Normalize to probability distributions
        p1 = psd1 / psd1.sum()
        p2 = psd2 / psd2.sum()
        kl_div = np.sum(p1 * np.log(p1 / p2))
        distance = kl_div
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return float(distance)


def batch_spectral_distance(
    predicted_psd: np.ndarray,
    reference_psd: np.ndarray,
    freqs: np.ndarray,
    metric: str = 'wasserstein'
) -> np.ndarray:
    """
    Compute spectral distance for batch
    
    Args:
        predicted_psd: Predicted PSDs [batch, n_freqs]
        reference_psd: Reference PSD [n_freqs] (expected from context)
        freqs: Frequency bins [n_freqs]
        metric: Distance metric
    
    Returns:
        distances: Spectral distances [batch]
    """
    if predicted_psd.ndim == 1:
        predicted_psd = predicted_psd[np.newaxis, :]
    
    batch_size = predicted_psd.shape[0]
    distances = np.zeros(batch_size)
    
    for i in range(batch_size):
        distances[i] = spectral_distance(
            predicted_psd[i],
            reference_psd,
            freqs,
            metric=metric
        )
    
    return distances


def visualize_psd(
    freqs: np.ndarray,
    psd_dict: Dict[str, np.ndarray],
    title: str = "Power Spectral Density",
    save_path: Optional[str] = None
):
    """
    Visualize PSDs for comparison
    
    Args:
        freqs: Frequency bins
        psd_dict: Dictionary of {label: psd}
        title: Plot title
        save_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    for label, psd in psd_dict.items():
        plt.semilogy(freqs, psd, label=label, linewidth=2)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 PSD plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Test PSD computation
    print("Testing PSD utilities...")
    
    # Generate test signals
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    
    # Signal 1: Low frequency (2 Hz)
    signal1 = np.sin(2 * np.pi * 2 * t) + 0.1 * np.random.randn(len(t))
    
    # Signal 2: High frequency (10 Hz)
    signal2 = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
    
    # Signal 3: Mixed
    signal3 = (np.sin(2 * np.pi * 2 * t) + 
               0.5 * np.sin(2 * np.pi * 10 * t) + 
               0.1 * np.random.randn(len(t)))
    
    print("\n1. Computing PSDs...")
    freqs1, psd1 = compute_psd(signal1, sampling_rate=100)
    freqs2, psd2 = compute_psd(signal2, sampling_rate=100)
    freqs3, psd3 = compute_psd(signal3, sampling_rate=100)
    
    print(f"   Signal 1 PSD: {psd1.shape}")
    print(f"   Signal 2 PSD: {psd2.shape}")
    print(f"   Signal 3 PSD: {psd3.shape}")
    
    # Test spectral distance
    print("\n2. Computing spectral distances...")
    dist_1_2 = spectral_distance(psd1, psd2, freqs1, metric='l1')
    dist_1_3 = spectral_distance(psd1, psd3, freqs1, metric='l1')
    dist_2_3 = spectral_distance(psd2, psd3, freqs2, metric='l1')
    
    print(f"   Distance(low_freq, high_freq): {dist_1_2:.4f}")
    print(f"   Distance(low_freq, mixed): {dist_1_3:.4f}")
    print(f"   Distance(high_freq, mixed): {dist_2_3:.4f}")
    
    # Test batch computation
    print("\n3. Testing batch computation...")
    batch_signals = np.stack([signal1, signal2, signal3])
    freqs_batch, psds_batch = compute_psd(batch_signals, sampling_rate=100)
    print(f"   Batch PSDs shape: {psds_batch.shape}")
    
    # Compute expected PSD
    freqs_exp, psd_expected = compute_expected_psd(batch_signals, sampling_rate=100)
    print(f"   Expected PSD shape: {psd_expected.shape}")
    
    # Batch distances
    distances = batch_spectral_distance(psds_batch, psd_expected, freqs_batch)
    print(f"   Batch distances: {distances}")
    
    print("\n✅ PSD utilities tests passed!")
    
    # Visualize
    print("\n📊 Creating visualization...")
    visualize_psd(
        freqs1,
        {
            'Low Freq (2 Hz)': psd1,
            'High Freq (10 Hz)': psd2,
            'Mixed': psd3
        },
        title="Test PSDs"
    )
    print("   (Close plot window to continue)")

