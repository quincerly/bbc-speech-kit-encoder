#!/usr/bin/env python3
"""
BBC Micro TMS5220 Speech Encoder — Auto-Optimising Version

Searches the encoding parameter space to find the combination that minimises
Log Spectral Distortion (LSD) between the original audio and the signal
synthesised by the TMS5220 chip model, subject to having zero beep frames.

Usage:
    beeb_speech_encode_opt input.wav output.ssd [--verbose]

The search covers:
    pre-emphasis      : 0.30, 0.50, 0.70, 0.97
    bandwidth exp.    : 0.980, 0.985, 0.990, 0.994
    voiced threshold  : 0.25, 0.30, 0.35, 0.40
    pitch ceiling     : 280, 300, 320 Hz
    voiced gap fill   : 1, 2, 3 frames

That is up to 4×4×4×3×3 = 576 combinations. In practice the inner loops
prune heavily; typical runtime is 20-60 seconds on a modern CPU.

Requires: numpy, scipy
"""

import sys
import wave
import numpy as np
from scipy import signal as scipy_signal

from scripts.encode import (
    TARGET_SR, FRAME_SAMPLES,
    ACORN_E_RAW, ACORN_P, ACORN_K, K_BITS,
    rev, quantise, energy_idx, pitch_idx,
    autocorr, levinson,
    fix_octave_errors, fill_voiced_gaps,
    bbcLine,
)
from scripts.synth import synthesise
from scripts.metrics import lsd, count_beep_frames


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse(orig, preemph, bwe, voiced_thresh, min_f0, max_f0, max_gap):
    """Run LPC analysis with the given parameters, return frame list."""
    emph = np.concatenate([[orig[0]], orig[1:] - preemph * orig[:-1]])
    total_frames = len(orig) // FRAME_SAMPLES
    frames = []
    for fi in range(total_frames):
        s = fi * FRAME_SAMPLES
        fr = orig[s:s + FRAME_SAMPLES]
        fe = emph[s:s + FRAME_SAMPLES]
        win = fe * np.hamming(FRAME_SAMPLES)
        e = float(np.sqrt(np.mean(fr ** 2)))

        min_lag = int(np.ceil(TARGET_SR / max_f0))
        max_lag = int(TARGET_SR / min_f0)
        r_p = autocorr(fr, max_lag)
        if r_p[0] > 1e-8:
            ns = r_p[min_lag:max_lag + 1] / r_p[0]
            bi = int(np.argmax(ns))
            p = float(TARGET_SR / (min_lag + bi)) if ns[bi] > voiced_thresh else 0.
        else:
            p = 0.

        r = autocorr(win, 10)
        k = levinson(r, 10)
        if bwe < 1.0:
            k = np.array([k[i] * (bwe ** (i + 1)) for i in range(10)])

        frames.append({'e': e, 'pitch': p, 'voiced': p > 0, 'k': k})

    pitches = [f['pitch'] for f in frames]
    fixed = fix_octave_errors(pitches)
    for fi, f in enumerate(frames):
        f['pitch'] = fixed[fi]
        f['voiced'] = fixed[fi] > 0
    frames = fill_voiced_gaps(frames, max_gap=max_gap)
    return frames


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_frames(frames):
    """Pack frames into a bit-reversed TMS5220 bitstream."""
    all_bits = []
    push = lambda v, n: [all_bits.append((v >> i) & 1) for i in range(n - 1, -1, -1)]

    for frame in frames:
        eI = energy_idx(frame['e'])
        if eI == 0:
            push(0, 4)
            continue
        push(eI, 4); push(0, 1)
        if frame['voiced']:
            push(pitch_idx(frame['pitch']), 6)
            for ki, (w, tbl) in enumerate(zip(K_BITS, ACORN_K)):
                push(quantise(frame['k'][ki] if ki < len(frame['k']) else 0, tbl), w)
        else:
            push(0, 6)
            for ki, (w, tbl) in enumerate(zip(K_BITS[:4], ACORN_K[:4])):
                push(quantise(frame['k'][ki] if ki < len(frame['k']) else 0, tbl), w)

    for _ in range(4):
        all_bits.append(1)
    while len(all_bits) % 8:
        all_bits.append(0)

    raw = bytes(
        sum(all_bits[i + j] << (7 - j) for j in range(8))
        for i in range(0, len(all_bits), 8)
    )
    return bytes(rev(b) for b in raw)


# ── Disk image builder ────────────────────────────────────────────────────────

def build_ssd(data_bytes, output_path):
    """Write a bootable BBC DFS .ssd disk image."""
    N = len(data_bytes)

    REM = 0xF4; READ = 0xF3; DIM = 0xDE; FOR = 0xE3; TO = 0xB8; NEXT = 0xED
    SOUND = 0xD4; AND_ = 0x80; REPEAT = 0xF5; UNTIL = 0xFD; CALL = 0xD6
    END = 0xE0; STEP = 0x88; DATA = 0xDC

    prog = b''
    prog += bbcLine(10,  REM,    ' BBC SPEECH PLAYER')
    prog += bbcLine(20,  READ,   ' N%')
    prog += bbcLine(30,  DIM,    ' A% N%')
    prog += bbcLine(40,  FOR,    'I%=0 ', TO, ' N%-1:', READ, ' B%:A%?I%=B%:', NEXT)
    prog += bbcLine(50,  SOUND,  ' &FF60,!A% ', AND_, ' &FFFF,0,0')
    prog += bbcLine(60,  FOR,    'I%=2 ', TO, ' N%-2 ', STEP, ' 2')
    prog += bbcLine(70,  SOUND,  ' &FF00,A%!I% ', AND_, ' &FFFF,0,0')
    prog += bbcLine(80,  NEXT)
    prog += bbcLine(90,  REPEAT, ':A%=&9E:', CALL, ' &FFF4:', UNTIL, ' (Y% ', AND_, ' &80)=0')
    prog += bbcLine(100, END)

    data_vals = [N] + list(data_bytes)
    dln = 1000
    prog += bbcLine(dln, DATA, ' ' + ','.join(str(v) for v in data_vals[:8])); dln += 10
    for i in range(8, len(data_vals), 8):
        prog += bbcLine(dln, DATA, ' ' + ','.join(str(v) for v in data_vals[i:i + 8]))
        dln += 10
    prog += bytes([0x0D, 0xFF])

    SECTOR = 256; NSECTORS = 800
    def sn(n): return (n + SECTOR - 1) // SECTOR
    boot = b'CHAIN "PLAYER"\r'
    psec = 2 + sn(len(boot))
    disk = bytearray(NSECTORS * SECTOR)
    disk[0:8]   = b'SPEECH  '
    disk[8:15]  = b'!BOOT  '; disk[15] = 0x24
    disk[16:23] = b'PLAYER '; disk[23] = 0x24
    disk[0x100:0x104] = b'    '
    disk[0x104] = 0; disk[0x105] = 0x10
    disk[0x106] = 0x30 | ((NSECTORS >> 8) & 3); disk[0x107] = NSECTORS & 0xFF

    def sf(n, load, exec_, length, start):
        b = 0x100 + n * 8
        disk[b]   = load   & 0xFF; disk[b+1] = (load   >> 8) & 0xFF
        disk[b+2] = exec_  & 0xFF; disk[b+3] = (exec_  >> 8) & 0xFF
        disk[b+4] = length & 0xFF; disk[b+5] = (length >> 8) & 0xFF
        disk[b+6] = (((start  >> 8) & 3) << 6 | ((length >> 16) & 3) << 4 |
                     ((exec_  >> 16) & 3) << 2 |  ((load   >> 16) & 3))
        disk[b+7] = start & 0xFF

    sf(1, 0, 0, len(boot), 2)
    sf(2, 0x1900, 0x8023, len(prog), psec)
    bs = bytearray(sn(len(boot)) * SECTOR); bs[:len(boot)] = boot
    disk[2 * SECTOR:2 * SECTOR + len(bs)] = bs
    ps = bytearray(sn(len(prog)) * SECTOR); ps[:len(prog)] = prog
    disk[psec * SECTOR:psec * SECTOR + len(ps)] = ps

    with open(output_path, 'wb') as f:
        f.write(bytes(disk))


# ── Parameter search ──────────────────────────────────────────────────────────

PARAM_GRID = {
    'preemph':      [0.30, 0.50, 0.70, 0.97],
    'bwe':          [0.980, 0.985, 0.990, 0.994],
    'voiced_thresh':[0.25, 0.30, 0.35, 0.40],
    'max_f0':       [280, 300, 320],
    'max_gap':      [1, 2, 3],
}

MIN_F0 = 75   # fixed — lower bound never changes


def optimise(orig, verbose=False):
    """
    Search the parameter grid and return (best_data_bytes, best_params, best_metrics).

    Strategy: for each combination, encode → synthesise → compute LSD.
    Reject any configuration with beep frames.
    Return the configuration with the lowest mean LSD.
    """
    best_lsd  = float('inf')
    best_data = None
    best_params = None
    best_metrics = None
    n_tried = 0
    n_skip_beep = 0

    total = (len(PARAM_GRID['preemph']) * len(PARAM_GRID['bwe']) *
             len(PARAM_GRID['voiced_thresh']) * len(PARAM_GRID['max_f0']) *
             len(PARAM_GRID['max_gap']))

    if verbose:
        print(f"Searching {total} parameter combinations...")
        print(f"{'preemph':>8} {'bwe':>6} {'vthresh':>8} {'max_f0':>7} {'gap':>4} "
              f"{'LSD_mn':>8} {'LSD_md':>8} {'beeps':>6}")

    for preemph in PARAM_GRID['preemph']:
        for bwe in PARAM_GRID['bwe']:
            for vthresh in PARAM_GRID['voiced_thresh']:
                for max_f0 in PARAM_GRID['max_f0']:
                    for max_gap in PARAM_GRID['max_gap']:
                        n_tried += 1

                        frames = analyse(orig, preemph, bwe, vthresh,
                                         MIN_F0, max_f0, max_gap)
                        data_bytes = encode_frames(frames)
                        synth_pcm  = synthesise(data_bytes)
                        beeps      = count_beep_frames(orig, synth_pcm)

                        if beeps > 0:
                            n_skip_beep += 1
                            if verbose:
                                print(f"{preemph:>8.2f} {bwe:>6.3f} {vthresh:>8.2f} "
                                      f"{max_f0:>7} {max_gap:>4}  "
                                      f"{'(beeps: '+str(beeps)+')':>20}")
                            continue

                        lsd_mean, lsd_med = lsd(orig, synth_pcm)

                        if verbose:
                            marker = ' <-- best' if lsd_mean < best_lsd else ''
                            print(f"{preemph:>8.2f} {bwe:>6.3f} {vthresh:>8.2f} "
                                  f"{max_f0:>7} {max_gap:>4} "
                                  f"{lsd_mean:>8.3f} {lsd_med:>8.3f} {beeps:>6}"
                                  f"{marker}")

                        if lsd_mean < best_lsd:
                            best_lsd   = lsd_mean
                            best_data  = data_bytes
                            best_params = {
                                'preemph': preemph, 'bwe': bwe,
                                'voiced_thresh': vthresh, 'max_f0': max_f0,
                                'max_gap': max_gap,
                            }
                            best_metrics = {'lsd_mean': lsd_mean, 'lsd_med': lsd_med}

    if verbose:
        print(f"\nSearched {n_tried} configs, skipped {n_skip_beep} with beeps.")

    return best_data, best_params, best_metrics


# ── CLI entry point ───────────────────────────────────────────────────────────

def main_cli():
    verbose = '--verbose' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('--')]

    if len(args) < 2:
        print("Usage: beeb_speech_encode_opt input.wav output.ssd [--verbose]")
        sys.exit(1)

    input_wav  = args[0]
    output_ssd = args[1]

    # Load and downsample
    with wave.open(input_wav, 'rb') as w:
        raw    = w.readframes(w.getnframes())
        src_sr = w.getframerate()
        ch     = w.getnchannels()
        sw     = w.getsampwidth()
    samples = np.frombuffer(raw, dtype=np.int16 if sw == 2 else np.int8).astype(np.float32)
    samples /= (32768. if sw == 2 else 128.)
    if ch == 2:
        samples = (samples[0::2] + samples[1::2]) / 2
    orig = scipy_signal.resample_poly(samples, TARGET_SR, src_sr)
    print(f"Loaded: {len(orig)/TARGET_SR:.2f}s  ({len(orig)//FRAME_SAMPLES} frames)")

    # Search
    print("Optimising encoding parameters (this takes ~30-60 seconds)...")
    best_data, best_params, best_metrics = optimise(orig, verbose=verbose)

    if best_data is None:
        print("ERROR: No valid encoding found (all configurations produced beep frames).")
        print("Try running beeb_speech_encode instead (uses fixed default parameters).")
        sys.exit(1)

    # Report
    print(f"\nBest parameters found:")
    for k, v in best_params.items():
        print(f"  {k:20s} = {v}")
    print(f"  LSD mean = {best_metrics['lsd_mean']:.3f} dB")
    print(f"  LSD median = {best_metrics['lsd_med']:.3f} dB")

    # Write disk image
    build_ssd(best_data, output_ssd)
    print(f"\nWritten: {output_ssd}")
    print("Boot with Shift+Break on BBC Model B with Speech System fitted.")


if __name__ == '__main__':
    main_cli()
