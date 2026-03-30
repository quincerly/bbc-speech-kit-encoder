#!/usr/bin/env python3
"""
BBC Micro TMS5220 Speech Encoder — Auto-Optimising Version

Searches the encoding parameter space to find the combination that minimises
LPC Spectral Distortion between the original audio frames and the
all-pole filter response of the quantised LPC parameters.

Usage:
    beeb_speech_encode_opt input.wav output.ssd [--verbose] [--workers N]

The search covers 576 combinations:
    pre-emphasis      : 0.30, 0.50, 0.70, 0.97
    bandwidth exp.    : 0.980, 0.985, 0.990, 0.994
    voiced threshold  : 0.25, 0.30, 0.35, 0.40
    pitch ceiling     : 280, 300, 320 Hz
    voiced gap fill   : 1, 2, 3 frames

Multiprocessing is used automatically — pass --workers N to override
the default (number of CPU cores).

Requires: numpy, scipy
"""

import sys
import os
import argparse
import itertools
import multiprocessing
import numpy as np

from scripts.encode import (
    TARGET_SR, FRAME_SAMPLES,
    ACORN_E_RAW, ACORN_P, ACORN_K, K_BITS,
    rev, quantise, energy_idx, pitch_idx,
    fix_octave_errors, fill_voiced_gaps,
    bbcLine, load_wav,
)
from scripts.synth import synthesise
from scripts.metrics import lpc_spectral_distortion


# ── Analysis ──────────────────────────────────────────────────────────────────


def _levinson_batch(R, p):
    """Vectorised Levinson-Durbin for N frames. R shape (N, p+1)."""
    N = R.shape[0]; A = np.zeros((N, p+1)); A[:,0] = 1.0
    K_out = np.zeros((N, p)); err = R[:,0].copy()
    for i in range(1, p+1):
        lam   = -np.einsum('nj,nj->n', A[:,:i], R[:,1:i+1][:,::-1])
        valid = err > 1e-15
        k     = np.where(valid, lam/(err+1e-30), 0.0)
        K_out[:,i-1] = k; A[:,i] = k
        tmp = A[:,:i].copy()
        for j in range(1, i): A[:,j] = tmp[:,j] + k*tmp[:,i-j]
        err *= np.where(valid, 1.0-k*k, 1.0); err = np.maximum(err, 0.0)
    return K_out


def analyse(orig, preemph, bwe, voiced_thresh, min_f0, max_f0, max_gap):
    """
    Fully vectorised LPC analysis — batch autocorr, pitch detection, and
    Levinson-Durbin across all frames simultaneously.
    ~3x faster than the per-frame version for long utterances.
    """
    nf      = len(orig) // FRAME_SAMPLES
    emph    = np.empty_like(orig)
    emph[0] = orig[0]; emph[1:] = orig[1:] - preemph * orig[:-1]

    min_lag = int(np.ceil(TARGET_SR / max_f0))
    max_lag = int(TARGET_SR / min_f0)
    win     = np.hamming(FRAME_SAMPLES)
    bwe_w   = np.array([bwe ** (i + 1) for i in range(10)])

    # Stack all frames into matrices (no Python loop)
    nfft = 1
    while nfft < 2 * FRAME_SAMPLES: nfft <<= 1
    idx_mat    = np.arange(nf)[:,None]*FRAME_SAMPLES + np.arange(FRAME_SAMPLES)[None,:]
    frames_mat = orig[idx_mat]                   # (nf, FRAME_SAMPLES)
    emph_mat   = emph[idx_mat] * win[None,:]     # (nf, FRAME_SAMPLES)

    # Batch FFT autocorrelation for pitch and LPC
    F1  = np.fft.rfft(frames_mat, n=nfft, axis=1)
    R1  = np.fft.irfft(F1 * np.conj(F1), n=nfft, axis=1)
    F2  = np.fft.rfft(emph_mat,   n=nfft, axis=1)
    R2  = np.fft.irfft(F2 * np.conj(F2), n=nfft, axis=1)

    energies = np.sqrt(np.mean(frames_mat**2, axis=1))   # (nf,)

    # Vectorised pitch detection
    r0s          = R1[:, 0]
    valid        = r0s > 1e-8
    pitch_window = R1[:, min_lag:max_lag+1]
    ns_all = np.where(valid[:,None], pitch_window / np.where(valid[:,None], r0s[:,None], 1.), 0.)
    bi_all = np.argmax(ns_all, axis=1)
    peak   = ns_all[np.arange(nf), bi_all]
    f0_all = np.where(valid & (peak > voiced_thresh),
                      TARGET_SR / (min_lag + bi_all).astype(float), 0.)

    # Batch Levinson-Durbin
    lpc_r = R2[:, :11]                                      # (nf, 11)
    K_all = _levinson_batch(lpc_r, 10) * bwe_w[None,:]      # (nf, 10)

    pitches = list(f0_all)
    fixed   = fix_octave_errors(pitches)
    frames  = [{'e': float(energies[fi]), 'pitch': fixed[fi],
                'voiced': fixed[fi] > 0, 'k': K_all[fi]}
               for fi in range(nf)]
    return fill_voiced_gaps(frames, max_gap=max_gap)


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_frames(frames):
    all_bits = []
    push = lambda v, n: [all_bits.append((v >> i) & 1) for i in range(n-1, -1, -1)]
    for frame in frames:
        eI = energy_idx(frame['e'])
        if eI == 0: push(0, 4); continue
        push(eI, 4); push(0, 1)
        if frame['voiced']:
            push(pitch_idx(frame['pitch']), 6)
            for ki, (w, tbl) in enumerate(zip(K_BITS, ACORN_K)):
                push(quantise(frame['k'][ki] if ki < len(frame['k']) else 0, tbl), w)
        else:
            push(0, 6)
            for ki, (w, tbl) in enumerate(zip(K_BITS[:4], ACORN_K[:4])):
                push(quantise(frame['k'][ki] if ki < len(frame['k']) else 0, tbl), w)
    for _ in range(4): all_bits.append(1)
    while len(all_bits) % 8: all_bits.append(0)
    raw = bytes(sum(all_bits[i+j] << (7-j) for j in range(8))
                for i in range(0, len(all_bits), 8))
    return bytes(rev(b) for b in raw)


# ── Disk image builder ────────────────────────────────────────────────────────

def build_ssd(data_bytes, output_path):
    N = len(data_bytes)
    REM=0xF4; READ=0xF3; DIM=0xDE; FOR=0xE3; TO=0xB8; NEXT=0xED
    SOUND=0xD4; AND_=0x80; REPEAT=0xF5; UNTIL=0xFD; CALL=0xD6
    END=0xE0; STEP=0x88; DATA=0xDC

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
    data_vals = [N] + list(data_bytes); dln = 1000
    prog += bbcLine(dln, DATA, ' ' + ','.join(str(v) for v in data_vals[:8])); dln += 10
    for i in range(8, len(data_vals), 8):
        prog += bbcLine(dln, DATA, ' ' + ','.join(str(v) for v in data_vals[i:i+8]))
        dln += 10
    prog += bytes([0x0D, 0xFF])

    SECTOR=256; NSECTORS=800
    def sn(n): return (n+SECTOR-1)//SECTOR
    boot=b'CHAIN "PLAYER"\r'; psec=2+sn(len(boot))
    disk=bytearray(NSECTORS*SECTOR)
    disk[0:8]=b'SPEECH  '; disk[8:15]=b'!BOOT  '; disk[15]=0x24
    disk[16:23]=b'PLAYER '; disk[23]=0x24
    disk[0x100:0x104]=b'    '; disk[0x104]=0; disk[0x105]=0x10
    disk[0x106]=0x30|((NSECTORS>>8)&3); disk[0x107]=NSECTORS&0xFF

    def sf(n, load, exec_, length, start):
        b=0x100+n*8
        disk[b]=load&0xFF; disk[b+1]=(load>>8)&0xFF
        disk[b+2]=exec_&0xFF; disk[b+3]=(exec_>>8)&0xFF
        disk[b+4]=length&0xFF; disk[b+5]=(length>>8)&0xFF
        disk[b+6]=(((start>>8)&3)<<6|((length>>16)&3)<<4|
                   ((exec_>>16)&3)<<2|((load>>16)&3))
        disk[b+7]=start&0xFF

    sf(1,0,0,len(boot),2); sf(2,0x1900,0x8023,len(prog),psec)
    bs=bytearray(sn(len(boot))*SECTOR); bs[:len(boot)]=boot
    disk[2*SECTOR:2*SECTOR+len(bs)]=bs
    ps=bytearray(sn(len(prog))*SECTOR); ps[:len(prog)]=prog
    disk[psec*SECTOR:psec*SECTOR+len(ps)]=ps
    with open(output_path,'wb') as f: f.write(bytes(disk))


# ── Worker (must be top-level for multiprocessing pickle) ─────────────────────

def _evaluate_config(args):
    """Evaluate one parameter combination. Returns (lsd_mean, lsd_med, params)."""
    orig, preemph, bwe, voiced_thresh, max_f0, max_gap = args
    frames   = analyse(orig, preemph, bwe, voiced_thresh, 75, max_f0, max_gap)
    lsd_mean, lsd_med = lpc_spectral_distortion(orig, frames)
    params   = dict(preemph=preemph, bwe=bwe, voiced_thresh=voiced_thresh,
                    max_f0=max_f0, max_gap=max_gap)
    return (lsd_mean, lsd_med, params)


# ── Parameter grid ────────────────────────────────────────────────────────────

PARAM_GRID = {
    'preemph':       [0.30, 0.50, 0.70, 0.97],
    'bwe':           [0.980, 0.985, 0.990, 0.994],
    'voiced_thresh': [0.25, 0.30, 0.35, 0.40],
    'max_f0':        [280, 300, 320],
    'max_gap':       [1, 2, 3],
}


def optimise(orig, n_workers=None, verbose=False):
    """Search the parameter grid in parallel."""
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()

    configs = list(itertools.product(
        PARAM_GRID['preemph'], PARAM_GRID['bwe'],
        PARAM_GRID['voiced_thresh'], PARAM_GRID['max_f0'], PARAM_GRID['max_gap'],
    ))
    tasks = [(orig, pe, bw, vt, mf, mg) for pe, bw, vt, mf, mg in configs]

    if verbose:
        print(f"Searching {len(tasks)} combinations "
              f"({n_workers} worker{'s' if n_workers>1 else ''})...")
        print(f"{'preemph':>8} {'bwe':>6} {'vthresh':>8} {'max_f0':>7} "
              f"{'gap':>4} {'LSD_mn':>8} {'LSD_md':>8}")

    best_lsd = float('inf'); best_params = None; best_metrics = None

    with multiprocessing.Pool(processes=n_workers) as pool:
        for result in pool.imap(_evaluate_config, tasks, chunksize=4):
            lsd_mean, lsd_med, params = result
            if verbose:
                m = ' <-- best' if lsd_mean < best_lsd else ''
                print(f"{params['preemph']:>8.2f} {params['bwe']:>6.3f} "
                      f"{params['voiced_thresh']:>8.2f} {params['max_f0']:>7} "
                      f"{params['max_gap']:>4} {lsd_mean:>8.3f} "
                      f"{lsd_med:>8.3f}{m}")
            if lsd_mean < best_lsd:
                best_lsd    = lsd_mean
                best_params = params
                best_metrics = {'lsd_mean': lsd_mean, 'lsd_med': lsd_med}

    if verbose:
        print(f"\nSearched {len(tasks)} configurations.")

    if best_params is None:
        return None, None, None

    # Re-encode with the winning parameters
    orig_arr = tasks[0][0]
    best_frames = analyse(orig_arr, best_params['preemph'], best_params['bwe'],
                          best_params['voiced_thresh'], 75,
                          best_params['max_f0'], best_params['max_gap'])
    best_data = encode_frames(best_frames)
    return best_data, best_params, best_metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def main_cli():
    parser = argparse.ArgumentParser(
        description='BBC Micro TMS5220 auto-optimising speech encoder')
    parser.add_argument('input',  help='Input WAV file')
    parser.add_argument('output', help='Output .ssd disk image')
    parser.add_argument('--verbose', action='store_true',
                        help='Print every combination tried')
    parser.add_argument('--workers', type=int, default=None,
                        help='Parallel workers (default: CPU count)')
    args = parser.parse_args()

    orig = load_wav(args.input)
    n = args.workers or multiprocessing.cpu_count()
    print(f"Loaded: {len(orig)/TARGET_SR:.2f}s  ({len(orig)//FRAME_SAMPLES} frames)")
    print(f"Optimising 576 combinations using {n} worker{'s' if n>1 else ''}...")

    best_data, best_params, best_metrics = optimise(
        orig, n_workers=n, verbose=args.verbose)

    if best_data is None:
        print("ERROR: No valid encoding found (all configs had beep frames).")
        sys.exit(1)

    print(f"\nBest parameters:")
    for k, v in best_params.items():
        print(f"  {k:20s} = {v}")
    print(f"  LSD mean   = {best_metrics['lsd_mean']:.3f} dB")
    print(f"  LSD median = {best_metrics['lsd_med']:.3f} dB")

    build_ssd(best_data, args.output)
    print(f"\nWritten: {args.output}")
    print("Boot with Shift+Break on BBC Model B with Speech System fitted.")


if __name__ == '__main__':
    main_cli()
