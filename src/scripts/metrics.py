"""
Speech quality metrics for comparing original and synthesised audio.

Log Spectral Distortion (LSD) is the primary metric — it measures the
average spectral difference frame-by-frame in dB, which correlates well
with perceptual quality for LPC-coded speech.
"""

import numpy as np


def lsd(original, synthesised, frame_samples=200, min_rms=0.005,
        nfft=256, eps=1e-10):
    """
    Log Spectral Distortion between original and synthesised signals.

    Computed per-frame over voiced/active speech regions (frames where the
    original RMS exceeds min_rms), then averaged.

    Returns (mean_lsd, median_lsd) in dB. Lower is better.
    A well-encoded TMS5220 utterance typically scores 9-11 dB.
    """
    n = min(len(original), len(synthesised))
    orig = original[:n]
    synth = synthesised[:n]

    total_frames = n // frame_samples
    lsd_vals = []

    for fi in range(total_frames):
        s = fi * frame_samples
        o_frame = orig[s:s + frame_samples]
        s_frame = synth[s:s + frame_samples]

        if np.sqrt(np.mean(o_frame ** 2)) < min_rms:
            continue

        win = np.hanning(frame_samples)
        o_spec = np.abs(np.fft.rfft(o_frame * win, n=nfft)) ** 2
        s_spec = np.abs(np.fft.rfft(s_frame * win, n=nfft)) ** 2

        log_diff = 10 * np.log10((o_spec + eps) / (s_spec + eps))
        lsd_vals.append(float(np.sqrt(np.mean(log_diff ** 2))))

    if not lsd_vals:
        return float('inf'), float('inf')

    return float(np.mean(lsd_vals)), float(np.median(lsd_vals))


def count_beep_frames(original, synthesised, frame_samples=200,
                      orig_min_rms=0.003, synth_threshold=0.50):
    """
    Count frames where the synthesised output is abnormally loud relative
    to the original — characteristic of near-unstable filter resonance ('beeps').

    A frame is flagged if:
      - the original RMS exceeds orig_min_rms (it's a speech frame, not silence)
      - the synthesised RMS exceeds synth_threshold (clipping or ringing)
    """
    n = min(len(original), len(synthesised))
    beeps = 0
    for fi in range(n // frame_samples):
        s = fi * frame_samples
        o_rms = float(np.sqrt(np.mean(original[s:s + frame_samples] ** 2)))
        s_rms = float(np.sqrt(np.mean(synthesised[s:s + frame_samples] ** 2)))
        if o_rms > orig_min_rms and s_rms > synth_threshold:
            beeps += 1
    return beeps
