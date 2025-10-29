import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

# Sampling frequency
fs = 1000
nyquist = fs / 2

def generate_ppg_signal(duration=10):
    """Generate synthetic PPG signal with realistic noise"""
    t = np.linspace(0, duration, int(fs * duration))
    
    # Clean PPG signal (heart rate ~75 bpm = 1.25 Hz)
    clean = np.sin(2 * np.pi * 1.25 * t) + 0.3 * np.sin(4 * np.pi * 1.25 * t)
    
    # Add realistic noise sources
    baseline_drift = 0.5 * np.sin(2 * np.pi * 0.25 * t)  # Breathing
    motion = 0.3 * np.sin(2 * np.pi * 0.1 * t)           # Motion artifacts
    powerline = 0.2 * np.sin(2 * np.pi * 50 * t)         # 50 Hz interference
    hf_noise = 0.1 * np.random.randn(len(t))             # High frequency noise
    
    noisy = clean + baseline_drift + motion + powerline + hf_noise
    
    return t, noisy, clean

def notch_filter(data, notch_freq=50, quality_factor=30):
    """Remove power-line interference (50/60 Hz)"""
    w0 = notch_freq / nyquist
    b, a = signal.iirnotch(w0, quality_factor)
    return signal.filtfilt(b, a, data)

def high_pass_filter(data, cutoff_freq=0.5, order=4):
    """Remove baseline drift"""
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype='high')
    return signal.filtfilt(b, a, data)

def low_pass_filter(data, cutoff_freq=10, order=4):
    """Remove high-frequency noise"""
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype='low')
    return signal.filtfilt(b, a, data)

def compute_spectrum(data):
    """Compute frequency spectrum using FFT"""
    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1/fs)
    
    positive_freq_idx = xf >= 0
    freqs = xf[positive_freq_idx]
    magnitude = 2.0/N * np.abs(yf[positive_freq_idx])
    
    return freqs, magnitude

def demonstrate_filtering():
    """Complete demonstration of PPG signal denoising"""
    
    # Generate noisy PPG signal
    t, noisy, clean = generate_ppg_signal(duration=10)
    
    # Apply cascaded filters
    step1 = notch_filter(noisy, notch_freq=50)           # Remove 50 Hz
    step2 = high_pass_filter(step1, cutoff_freq=0.5)     # Remove baseline drift
    filtered = low_pass_filter(step2, cutoff_freq=10)    # Remove HF noise
    
    # Compute frequency spectra
    freq_noisy, mag_noisy = compute_spectrum(noisy)
    freq_filtered, mag_filtered = compute_spectrum(filtered)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Time domain - Noisy
    axes[0, 0].plot(t[:2000], noisy[:2000], 'r-', alpha=0.7, linewidth=0.8)
    axes[0, 0].set_title('Noisy PPG Signal', fontweight='bold')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Frequency domain - Noisy
    axes[0, 1].plot(freq_noisy, mag_noisy, 'r-', linewidth=1)
    axes[0, 1].axvline(x=50, color='orange', linestyle='--', label='50 Hz Power-line')
    axes[0, 1].set_title('Noisy Frequency Spectrum', fontweight='bold')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].set_xlim([0, 100])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Time domain - Comparison
    axes[1, 0].plot(t[:2000], noisy[:2000], 'r-', alpha=0.4, linewidth=0.8, label='Noisy')
    axes[1, 0].plot(t[:2000], filtered[:2000], 'g-', linewidth=1.5, label='Filtered')
    axes[1, 0].plot(t[:2000], clean[:2000], 'b--', linewidth=1, label='Original Clean')
    axes[1, 0].set_title('Denoising Result', fontweight='bold')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Frequency domain - Comparison
    axes[1, 1].plot(freq_noisy, mag_noisy, 'r-', alpha=0.4, linewidth=1, label='Noisy')
    axes[1, 1].plot(freq_filtered, mag_filtered, 'g-', linewidth=2, label='Filtered')
    axes[1, 1].set_title('Filtered Frequency Spectrum', fontweight='bold')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].set_xlim([0, 100])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate SNR improvement
    noise_before = noisy - clean
    noise_after = filtered - clean
    snr_before = 10 * np.log10(np.var(clean) / np.var(noise_before))
    snr_after = 10 * np.log10(np.var(clean) / np.var(noise_after))
    
    print(f"\n{'='*50}")
    print(f"PPG Signal Denoising Results")
    print(f"{'='*50}")
    print(f"Filters Applied:")
    print(f"  1. Notch Filter (50 Hz)")
    print(f"  2. High-Pass Filter (0.5 Hz)")
    print(f"  3. Low-Pass Filter (10 Hz)")
    print(f"\nSNR Before: {snr_before:.2f} dB")
    print(f"SNR After:  {snr_after:.2f} dB")
    print(f"Improvement: {snr_after - snr_before:.2f} dB")
    print(f"{'='*50}\n")


# Run the demonstration
if __name__ == "__main__":
    demonstrate_filtering()