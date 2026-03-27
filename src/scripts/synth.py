"""
TMS5220 LPC synthesiser using the Acorn Appendix D parameter tables.

Uses the same tables as the encoder so that synthesis accurately reflects
what the chip will produce when driven with our encoded data.
"""

import numpy as np
from scripts.encode import (
    ACORN_E_RAW, ACORN_P, ACORN_K, K_BITS,
    rev, quantise, energy_idx, pitch_idx,
)

# BeebEm 4.03 chirp table (41 signed bytes, from speech.cpp)
CHIRP = [
    0x00, 0x2a, -0x2c, 0x32,
   -0x4e, 0x12, 0x25, 0x14,
    0x02, -0x1f, -0x3b, 0x02,
    0x5f, 0x5a, 0x05, 0x0f,
    0x26, -0x04, -0x5b, -0x5b,
   -0x2a, -0x23, -0x24, -0x04,
    0x25, 0x2b, 0x22, 0x21,
    0x0f, -0x01, -0x08, -0x12,
   -0x13, -0x11, -0x09, -0x0a,
   -0x06,  0x00,  0x03,  0x02,
    0x01,
]

FRAME_SAMPLES = 200
SAMPLE_RATE   = 8000


def _k_to_predictor(k_vals):
    """PARCOR → direct-form predictor (Levinson forward step)."""
    p = len(k_vals)
    a = np.zeros(p)
    for i in range(p):
        a[i] = k_vals[i]
        for j in range(i // 2 + (i % 2)):
            tmp      = a[j] + k_vals[i] * a[i-1-j]
            a[i-1-j] = a[i-1-j] + k_vals[i] * a[j]
            a[j]     = tmp
    return a


def synthesise(encoded_bytes):
    """
    Synthesise PCM from a TMS5220 bitstream (bit-reversed bytes, as in DATA statements).
    Returns float32 numpy array normalised to ±1, at 8000 Hz.
    """
    # Undo the byte bit-reversal, then unpack MSB-first
    def unrev(b):
        r = 0
        for _ in range(8): r = (r << 1) | (b & 1); b >>= 1
        return r

    bits = []
    for byte in encoded_bytes:
        b = unrev(byte)
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)

    def read_bits(pos, n):
        v = 0
        for i in range(n):
            if pos + i < len(bits):
                v = (v << 1) | bits[pos + i]
        return v, pos + n

    # Decode frames using the same Acorn tables the encoder used
    frames = []
    pos = 0
    while pos + 4 <= len(bits):
        eI, pos = read_bits(pos, 4)
        if eI == 0:
            frames.append({'eI': 0, 'pI': 0, 'voiced': False, 'k': np.zeros(10)})
            continue
        if eI == 15:
            break
        rep, pos = read_bits(pos, 1)
        pI,  pos = read_bits(pos, 6)
        voiced = pI > 0

        if rep:
            prev_k = frames[-1]['k'] if frames else np.zeros(10)
            frames.append({'eI': eI, 'pI': pI, 'voiced': voiced, 'k': prev_k.copy()})
        else:
            widths = K_BITS if voiced else K_BITS[:4]
            k = np.zeros(10)
            for ki, w in enumerate(widths):
                idx, pos = read_bits(pos, w)
                k[ki] = ACORN_K[ki][idx]
            if not voiced:
                k[4:] = 0.0
            frames.append({'eI': eI, 'pI': pI, 'voiced': voiced, 'k': k})

    # Synthesise
    pcm = []
    x = np.zeros(10)
    rng = 0x1FFF
    pc  = 0.0
    cur_e = 0.0; cur_p = 0.0; cur_k = np.zeros(10)

    for frame in frames:
        tgt_e = float(ACORN_E_RAW[frame['eI']])
        tgt_p = float(ACORN_P[frame['pI']])
        tgt_k = frame['k'].copy()
        if not frame['voiced']:
            cur_k[4:] = 0.0; tgt_k[4:] = 0.0

        for period in range(8):
            frac = (period + 1) / 8.0
            ie = cur_e + (tgt_e - cur_e) * frac
            ip = cur_p + (tgt_p - cur_p) * frac
            ik = cur_k + (tgt_k - cur_k) * frac

            a    = _k_to_predictor(ik)
            gain = int(ie) >> 6

            for _ in range(25):
                if frame['voiced'] and ip > 0:
                    ci  = int(pc)
                    exc = (CHIRP[ci] * gain) // 256 if ci < len(CHIRP) else 0
                    pc += 1.0
                    if pc >= ip: pc = 0.0
                elif frame['eI'] > 0:
                    bit = ((rng >> 12) ^ rng) & 1
                    rng = ((rng << 1) | bit) & 0x1FFF
                    exc = gain if bit else -gain
                    pc  = 0.0
                else:
                    exc = 0; pc = 0.0

                y = exc - float(np.dot(a, x))
                y = max(-8192.0, min(8191.0, y))
                x = np.roll(x, 1); x[0] = y
                pcm.append(y / 8192.0)

        cur_e = tgt_e; cur_p = tgt_p; cur_k = tgt_k.copy()

    return np.array(pcm, dtype=np.float32)
