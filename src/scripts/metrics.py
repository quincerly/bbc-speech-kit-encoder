"""
Speech quality metric for LPC encoder optimisation.

lpc_spectral_distortion() compares the spectral envelope of each original
audio frame to the all-pole LPC filter response of the quantised parameters.
This avoids depending on the Python synthesiser and focuses purely on how
well the encoded LPC filter captures the original spectral shape.

Fully vectorised: batch FFT across all frames, vectorised Levinson-Durbin.
~30ms for a 463-frame (11.6s) utterance.
"""

import numpy as np
from scripts.encode import ACORN_K, K_BITS, quantise, energy_idx

# Pre-built sorted numpy arrays for fast vectorised quantisation
_K_TABLES = [np.array(tbl) for tbl in ACORN_K]


def _fast_quantise_batch(vals, tbl):
    """Vectorised nearest-neighbour lookup into tbl (must be sorted)."""
    idx      = np.searchsorted(tbl, vals)
    idx      = np.clip(idx, 0, len(tbl) - 1)
    idx_left = np.clip(idx - 1, 0, len(tbl) - 1)
    closer   = np.abs(tbl[idx_left] - vals) < np.abs(tbl[idx] - vals)
    return tbl[np.where(closer, idx_left, idx)]


def _levinson_batch(R, p):
    """
    Vectorised Levinson-Durbin for N frames simultaneously.
    R shape (N, p+1): autocorrelation lags 0..p.
    Returns K shape (N, p): PARCOR reflection coefficients.
    """
    N = R.shape[0]
    A = np.zeros((N, p + 1)); A[:, 0] = 1.0
    K_out = np.zeros((N, p))
    err = R[:, 0].copy()
    for i in range(1, p + 1):
        lam   = -np.einsum('nj,nj->n', A[:, :i], R[:, 1:i+1][:, ::-1])
        valid = err > 1e-15
        k     = np.where(valid, lam / (err + 1e-30), 0.0)
        K_out[:, i-1] = k
        A[:, i] = k
        tmp = A[:, :i].copy()
        for j in range(1, i):
            A[:, j] = tmp[:, j] + k * tmp[:, i-j]
        err *= np.where(valid, 1.0 - k * k, 1.0)
        err  = np.maximum(err, 0.0)
    return K_out


def _parcor_to_predictor_batch(K):
    """
    Vectorised PARCOR → direct-form predictor for N frames.
    K shape (N, 10). Returns A shape (N, 10).
    """
    A = K.copy()
    for i in range(10):
        A[:, i] = K[:, i]
        for j in range(i // 2 + (i % 2)):
            tmp        = A[:, j].copy()
            A[:, j]    = tmp + K[:, i] * A[:, i-1-j]
            A[:, i-1-j] = A[:, i-1-j] + K[:, i] * tmp
    return A


def lpc_spectral_distortion(original, frames, frame_samples=200,
                             min_rms=0.005, nfft=512):
    """
    LPC Spectral Distortion between original frame spectra and the
    all-pole LPC filter response of the quantised parameters.

    For each active frame:
      - Hanning-windowed power spectrum of the original audio
      - 1/|A(f)|^2 where A(z) is the quantised LPC predictor polynomial
      - both normalised to unit mean energy (shape comparison, not amplitude)
      - RMS of the log-spectral difference gives per-frame distortion

    Fully vectorised: batch FFT + vectorised Levinson + vectorised k quantisation.

    Returns (mean_lsd, median_lsd) in dB. Lower is better.
    """
    nf  = len(original) // frame_samples
    win = np.hanning(frame_samples)
    eps = 1e-10

    # Per-frame quantities
    nframes = min(nf, len(frames))
    idx_mat = (np.arange(nframes)[:, None] * frame_samples
               + np.arange(frame_samples)[None, :])
    frames_mat = original[idx_mat]                    # (nframes, frame_samples)

    energies = np.sqrt(np.mean(frames_mat ** 2, axis=1))
    eI_all   = np.array([energy_idx(frames[fi]['e']) for fi in range(nframes)])
    active   = (energies >= min_rms) & (eI_all > 0)

    if not active.any():
        return float('inf'), float('inf')

    act_idx = np.where(active)[0]
    N = len(act_idx)

    # Vectorised k quantisation
    f_v = np.array([frames[fi]['voiced'] for fi in range(nframes)])
    f_k = np.array([frames[fi]['k'] for fi in range(nframes)])  # (nframes, 10)

    k_q = np.zeros_like(f_k)
    for ki, tbl in enumerate(_K_TABLES):
        k_q[:, ki] = _fast_quantise_batch(f_k[:, ki], tbl)
    k_q[~f_v, 4:] = 0.0   # zero K5-K10 for unvoiced frames

    # Only active frames
    K_act = k_q[active]   # (N, 10)

    # Vectorised Levinson predictor conversion
    A = _parcor_to_predictor_batch(K_act)  # (N, 10)

    # Build predictor polynomial matrices
    poly = np.zeros((N, nfft))
    poly[:, 0]    = 1.0
    poly[:, 1:11] = A

    # Windowed original frames
    O_pad = np.zeros((N, nfft))
    O_pad[:, :frame_samples] = frames_mat[act_idx] * win[None, :]

    # Batch FFT
    orig_spec = np.abs(np.fft.rfft(O_pad, axis=1)) ** 2   # (N, nfft/2+1)
    A_spec    = np.abs(np.fft.rfft(poly,  axis=1)) ** 2
    lpc_spec  = 1.0 / (A_spec + eps)

    # Normalise each frame to unit mean energy
    orig_norm = orig_spec / (np.mean(orig_spec, axis=1, keepdims=True) + eps)
    lpc_norm  = lpc_spec  / (np.mean(lpc_spec,  axis=1, keepdims=True) + eps)

    log_diff = 10.0 * np.log10((lpc_norm + eps) / (orig_norm + eps))
    lsd_vals = np.sqrt(np.mean(log_diff ** 2, axis=1))

    return float(np.mean(lsd_vals)), float(np.median(lsd_vals))


# Backward-compatible alias
def lsd(original, synthesised, frame_samples=200, min_rms=0.005, nfft=256):
    """Legacy synthesis-based LSD. Prefer lpc_spectral_distortion()."""
    n   = min(len(original), len(synthesised))
    eps = 1e-10; win = np.hanning(frame_samples); lsd_vals = []
    for fi in range(n // frame_samples):
        s = fi * frame_samples
        o_f = original[s:s+frame_samples]; s_f = synthesised[s:s+frame_samples]
        if np.sqrt(np.mean(o_f**2)) < min_rms: continue
        o_spec = np.abs(np.fft.rfft(o_f*win, n=nfft))**2
        s_spec = np.abs(np.fft.rfft(s_f*win, n=nfft))**2
        log_diff = 10.0*np.log10((o_spec+eps)/(s_spec+eps))
        lsd_vals.append(float(np.sqrt(np.mean(log_diff**2))))
    if not lsd_vals: return float('inf'), float('inf')
    return float(np.mean(lsd_vals)), float(np.median(lsd_vals))
