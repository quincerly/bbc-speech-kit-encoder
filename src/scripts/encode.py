#!/usr/bin/env python3
"""
BBC Micro TMS5220 Speech Encoder
Usage: python3 encode.py input.wav output.ssd
Requires: numpy, scipy
"""

import sys
import wave
import numpy as np
from scipy import signal as scipy_signal

TARGET_SR = 8000
FRAME_SAMPLES = 200

# Energy table — chip ROM values from MAME/BeebEm (energytable[i] >> 6).
# Index 0 = silence, index 15 = stop frame.
# These are the actual TMS5220 gain values used during synthesis.
# The Acorn Appendix D table is a ~17x scaled version of the same values
# and gives the same quantisation indices, but using the chip values here
# keeps encoding and synthesis on a consistent scale.
_MAME_ENERGY_RAW = [0x0000,0x00C0,0x0140,0x01C0,0x0280,0x0380,0x0500,0x0740,
                    0x0A00,0x0E40,0x1440,0x1C80,0x2840,0x38C0,0x5040,0x7FC0]
ACORN_E_RAW = np.array([v >> 6 for v in _MAME_ENERGY_RAW])
# Pitch period table — values are sample periods at 8kHz (index 0 = unvoiced).
# These are the actual TMS5220 chip ROM values from MAME/BeebEm, derived by
# decapping the chip. The Acorn Speech System User Guide (Appendix D) documents
# a slightly different table; the chip ROM values are used here as they reflect
# what the hardware actually does.
ACORN_P = [0,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
           31,32,33,34,35,36,37,38,39,40,41,42,43,45,47,49,
           51,53,54,57,59,61,63,66,69,71,73,77,79,81,85,87,
           92,95,99,102,106,110,115,119,123,128,133,138,143,149,154,160]
ACORN_K = [
    np.array([-501,-498,-497,-495,-493,-491,-488,-482,-478,-474,-469,-464,-459,-452,-445,-437,
              -412,-380,-339,-288,-227,-158,-81,-1,80,157,226,287,337,379,411,436])/512.,
    np.array([-328,-303,-274,-244,-211,-175,-138,-99,-59,-18,24,64,105,143,180,215,
               248,278,306,331,354,374,392,408,422,435,445,455,463,470,476,506])/512.,
    np.array([-441,-387,-333,-279,-217,-171,-117,-63,-9,45,98,152,206,260,314,368])/512.,
    np.array([-328,-273,-217,-161,-106,-50,5,61,116,172,228,283,339,394,450,506])/512.,
    np.array([-328,-282,-235,-189,-142,-96,-50,-3,43,90,136,182,229,275,322,368])/512.,
    np.array([-256,-212,-168,-123,-79,-35,10,54,98,143,187,232,276,320,365,409])/512.,
    np.array([-308,-260,-212,-164,-117,-69,-21,27,75,122,170,218,266,314,361,409])/512.,
    np.array([-256,-161,-66,29,124,219,314,409])/512.,
    np.array([-256,-176,-96,-15,65,146,226,307])/512.,
    np.array([-205,-132,-59,14,87,160,234,307])/512.,
]
K_BITS = [5,5,4,4,4,4,4,3,3,3]

def rev(b):
    r = 0
    for i in range(8): r = (r<<1)|(b&1); b >>= 1
    return r

def quantise(v, t): return int(np.argmin(np.abs(t-v)))

def energy_idx(e):
    # Scale RMS to MAME gain range (0-511 = 9-bit).
    # The TMS5220 encodes energy on a roughly log scale;
    # we find the table entry closest to e * 512.
    e_scaled = e * 512.0
    if e_scaled < ACORN_E_RAW[1]: return 0
    return min(range(1, 15), key=lambda i: abs(e_scaled - ACORN_E_RAW[i]))

def pitch_idx(f0):
    if f0 <= 0: return 0
    return 1+int(np.argmin(np.abs(np.array(ACORN_P[1:])-8000/f0)))

def autocorr(d, ml):
    n = len(d); r = np.zeros(ml+1)
    for lag in range(ml+1): r[lag] = np.dot(d[:n-lag], d[lag:])
    return r

def levinson(r, p):
    a = np.zeros(p+1); a[0] = 1.; kOut = np.zeros(p); err = r[0]
    if err <= 0: return kOut
    for i in range(1, p+1):
        lam = -np.dot(a[:i], r[1:i+1][::-1])
        if err < 1e-15: break
        k = lam/err; kOut[i-1] = k; a[i] = k
        tmp = a.copy()
        for j in range(1,i): tmp[j] = a[j]+k*a[i-j]
        a[:i+1] = tmp[:i+1]; err *= (1-k*k)
        if err <= 0: break
    return kOut

def fix_octave_errors(pitches, max_ratio=1.7):
    p = np.array(pitches, dtype=float); n = len(p); fixed = p.copy()
    for i in range(1, n-1):
        if p[i] == 0: continue
        neighbours = [p[j] for j in range(max(0,i-2), min(n,i+3)) if j!=i and p[j]>0]
        if not neighbours: continue
        med = np.median(neighbours); ratio = p[i]/med
        if med > 0 and ratio > max_ratio: fixed[i] = p[i]/2
        elif med > 0 and ratio < 1/max_ratio: fixed[i] = p[i]*2
    return list(fixed)

def fill_voiced_gaps(frames, max_gap=2):
    n = len(frames); result = [dict(f) for f in frames]
    i = 0
    while i < n:
        if result[i]['voiced']: i += 1; continue
        j = i
        while j < n and not result[j]['voiced']: j += 1
        gap = j-i
        if gap <= max_gap and i > 0 and j < n:
            p_before = result[i-1]['pitch']; p_after = result[j]['pitch']
            for k in range(gap):
                frac = (k+1)/(gap+1)
                result[i+k]['pitch'] = p_before*(1-frac)+p_after*frac
                result[i+k]['voiced'] = True
        i = j
    return result

def bbcLine(num, *parts):
    body = []
    for p in parts:
        if isinstance(p, int): body.append(p)
        else: body.extend(ord(c) for c in p)
    return bytes([0x0D, (num>>8)&0xFF, num&0xFF, 4+len(body)] + body)


def load_wav(input_wav):
    """Load a WAV file and return a float32 mono array at 8kHz."""
    with wave.open(input_wav, 'rb') as w:
        raw = w.readframes(w.getnframes())
        src_sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
    samples = np.frombuffer(raw, dtype=np.int16 if sw==2 else np.int8).astype(np.float32)
    samples /= (32768. if sw==2 else 128.)
    if ch == 2: samples = (samples[0::2]+samples[1::2])/2
    return scipy_signal.resample_poly(samples, TARGET_SR, src_sr)


def encode_to_ssd(orig, output_ssd,
                  preemph=0.30, bwe=0.985, voiced_thresh=0.30,
                  min_f0=75, max_f0=300, max_gap=2):
    """Full pipeline: numpy audio array → .ssd disk image file."""
    emph = np.concatenate([[orig[0]], orig[1:]-preemph*orig[:-1]])
    total_frames = len(orig)//FRAME_SAMPLES
    frames = []
    for fi in range(total_frames):
        s = fi*FRAME_SAMPLES
        fr = orig[s:s+FRAME_SAMPLES]; fe = emph[s:s+FRAME_SAMPLES]
        win = fe*np.hamming(FRAME_SAMPLES)
        e = float(np.sqrt(np.mean(fr**2)))
        min_lag = int(np.ceil(TARGET_SR/max_f0)); max_lag = int(TARGET_SR/min_f0)
        r_p = autocorr(fr, max_lag)
        if r_p[0] > 1e-8:
            ns = r_p[min_lag:max_lag+1]/r_p[0]; bi = int(np.argmax(ns))
            p = float(TARGET_SR/(min_lag+bi)) if ns[bi]>voiced_thresh else 0.
        else: p = 0.
        r = autocorr(win, 10); k = levinson(r, 10)
        k = np.array([k[i]*(bwe**(i+1)) for i in range(10)])
        frames.append({'e':e, 'pitch':p, 'voiced':p>0, 'k':k})

    pitches = [f['pitch'] for f in frames]
    fixed = fix_octave_errors(pitches)
    for fi,f in enumerate(frames): f['pitch'] = fixed[fi]; f['voiced'] = fixed[fi]>0
    frames = fill_voiced_gaps(frames, max_gap=max_gap)

    all_bits = []; push = lambda v,n: [all_bits.append((v>>i)&1) for i in range(n-1,-1,-1)]
    for frame in frames:
        eI = energy_idx(frame['e'])
        if eI == 0: push(0,4); continue
        push(eI,4); push(0,1)
        if frame['voiced']:
            push(pitch_idx(frame['pitch']), 6)
            for ki,(w,tbl) in enumerate(zip(K_BITS, ACORN_K)):
                push(quantise(frame['k'][ki] if ki<len(frame['k']) else 0, tbl), w)
        else:
            push(0, 6)
            for ki,(w,tbl) in enumerate(zip(K_BITS[:4], ACORN_K[:4])):
                push(quantise(frame['k'][ki] if ki<len(frame['k']) else 0, tbl), w)
    for _ in range(4): all_bits.append(1)
    while len(all_bits)%8: all_bits.append(0)
    raw_bytes = bytes(sum(all_bits[i+j]<<(7-j) for j in range(8)) for i in range(0,len(all_bits),8))
    data_bytes = bytes(rev(b) for b in raw_bytes)
    N = len(data_bytes)

    REM=0xF4; READ=0xF3; DIM=0xDE; FOR=0xE3; TO=0xB8; NEXT=0xED
    SOUND=0xD4; AND_=0x80; REPEAT=0xF5; UNTIL=0xFD; CALL=0xD6; END=0xE0
    STEP=0x88; DATA=0xDC

    prog = b''
    prog += bbcLine(10,  REM,  ' BBC SPEECH PLAYER')
    prog += bbcLine(20,  READ, ' N%')
    prog += bbcLine(30,  DIM,  ' A% N%')
    prog += bbcLine(40,  FOR,  'I%=0 ', TO, ' N%-1:', READ, ' B%:A%?I%=B%:', NEXT)
    prog += bbcLine(50,  SOUND,' &FF60,!A% ', AND_, ' &FFFF,0,0')
    prog += bbcLine(60,  FOR,  'I%=2 ', TO, ' N%-2 ', STEP, ' 2')
    prog += bbcLine(70,  SOUND,' &FF00,A%!I% ', AND_, ' &FFFF,0,0')
    prog += bbcLine(80,  NEXT)
    prog += bbcLine(90,  REPEAT, ':A%=&9E:', CALL, ' &FFF4:', UNTIL, ' (Y% ', AND_, ' &80)=0')
    prog += bbcLine(100, END)
    data_vals = [N]+list(data_bytes); dln = 1000
    prog += bbcLine(dln, DATA, ' '+','.join(str(v) for v in data_vals[:8])); dln += 10
    for i in range(8, len(data_vals), 8):
        prog += bbcLine(dln, DATA, ' '+','.join(str(v) for v in data_vals[i:i+8])); dln += 10
    prog += bytes([0x0D, 0xFF])

    SECTOR=256; NSECTORS=800
    def sn(n): return (n+SECTOR-1)//SECTOR
    boot = b'CHAIN "PLAYER"\r'; psec = 2+sn(len(boot))
    disk = bytearray(NSECTORS*SECTOR)
    disk[0:8]=b'SPEECH  '; disk[8:15]=b'!BOOT  '; disk[15]=0x24
    disk[16:23]=b'PLAYER '; disk[23]=0x24
    disk[0x100:0x104]=b'    '; disk[0x104]=0; disk[0x105]=0x10
    disk[0x106]=0x30|((NSECTORS>>8)&3); disk[0x107]=NSECTORS&0xFF

    def sf(n, load, exec_, length, start):
        b = 0x100+n*8
        disk[b]=load&0xFF; disk[b+1]=(load>>8)&0xFF
        disk[b+2]=exec_&0xFF; disk[b+3]=(exec_>>8)&0xFF
        disk[b+4]=length&0xFF; disk[b+5]=(length>>8)&0xFF
        disk[b+6]=((start>>8)&3)<<6|((length>>16)&3)<<4|((exec_>>16)&3)<<2|((load>>16)&3)
        disk[b+7]=start&0xFF

    sf(1, 0, 0, len(boot), 2)
    sf(2, 0x1900, 0x8023, len(prog), psec)
    bs = bytearray(sn(len(boot))*SECTOR); bs[:len(boot)] = boot
    disk[2*SECTOR:2*SECTOR+len(bs)] = bs
    ps = bytearray(sn(len(prog))*SECTOR); ps[:len(prog)] = prog
    disk[psec*SECTOR:psec*SECTOR+len(ps)] = ps

    with open(output_ssd, 'wb') as f: f.write(bytes(disk))
    return N, len(frames), sum(1 for f in frames if f['voiced'])


def main_cli():
    if len(sys.argv) < 3:
        print("Usage: beeb_speech_encode input.wav output.ssd")
        sys.exit(1)
    input_wav  = sys.argv[1]
    output_ssd = sys.argv[2]
    orig = load_wav(input_wav)
    print(f"Loaded: {len(orig)/TARGET_SR:.2f}s  ({len(orig)//FRAME_SAMPLES} frames)")
    N, nf, nv = encode_to_ssd(orig, output_ssd)
    print(f"Voiced: {nv}/{nf}  Encoded: {N} bytes")
    print(f"Written: {output_ssd}")
    print("Boot with Shift+Break on BBC Model B with Speech System fitted.")


if __name__ == '__main__':
    main_cli()
