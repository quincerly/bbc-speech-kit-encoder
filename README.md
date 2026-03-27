# BBC Micro TMS5220 Speech Encoder

Encodes arbitrary WAV audio into a bootable BBC Model B DFS disk image (`.ssd`)
for playback via the Acorn Speech System (TMS5220 chip).

## Requirements


```
pip install <bbc-speech-kit-encoder dir>
```

## Usage

```
beeb_speech_encode input.wav output.ssd
```

The resulting `.ssd` file boots with **Shift+Break** on a BBC Model B with the
Speech System cartridge fitted. It chains a BBC BASIC program (`PLAYER`) which
streams the LPC data to the chip via `SOUND &FF60` / `SOUND &FF00`.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--preemph` | 0.30 | Pre-emphasis coefficient applied before LPC analysis |
| `--vthresh` | 0.30 | Voiced/unvoiced detection threshold (normalised autocorr peak) |
| `--min-f0` | 75 | Minimum pitch frequency (Hz) |
| `--max-f0` | 300 | Maximum pitch frequency (Hz) |
| `--gap` | 2 | Maximum unvoiced gap (frames) to fill by interpolation |
| `--bwe` | 0.985 | Bandwidth expansion factor — moves LPC poles away from unit circle |
| `--window` | hamming | Analysis window: `hamming`, `hanning`, `blackman` |
| `--no-octave-fix` | off | Disable octave error correction on the pitch track |

## How it works

1. **Downsample** — input WAV is resampled to 8 kHz (the TMS5220's native rate)
2. **Pre-emphasis** — first-order high-pass filter `y[n] = x[n] - α·x[n-1]`
3. **LPC analysis** — 10th-order Levinson-Durbin on 25 ms Hamming-windowed frames
4. **Bandwidth expansion** — reflection coefficients scaled by `λ^i` to prevent
   near-unit-circle poles from producing ringing artefacts
5. **Pitch detection** — normalised autocorrelation with octave-error correction
   and short unvoiced gap filling
6. **Quantisation** — parameters mapped to Acorn's Appendix D tables
7. **Bit packing** — MSB-first into bytes, then each byte bit-reversed for the
   OS `SOUND` interface (which sends bytes LSB-first to the chip)
8. **DFS image** — `$.!BOOT` + tokenised `$.PLAYER` written into a 400 KB
   single-sided 40-track Acorn DFS `.ssd`

## Parameter tables

All LPC parameter tables are taken directly from **Appendix D of the Acorn
Speech System User Guide** — the same values burned into the TMS5220's internal
ROMs on the Acorn Speech cartridge. The energy table has 16 entries (index 15 =
stop frame), pitch has 64 entries (index 0 = unvoiced), K1–K2 have 32 entries
each (5 bits), K3–K7 have 16 entries (4 bits), K8–K10 have 8 entries (3 bits).

## Bit ordering

The OS `SOUND &FF60` / `SOUND &FF00` interface sends each byte to the TMS5220
chip, which reads bits **LSB-first**. The LPC bitstream is packed **MSB-first**
into bytes (so the first parameter bit is bit 7 of byte 0). Each byte must
therefore be **bit-reversed** before storage in the `DATA` statements. This is
equivalent to what PROCSPEAK's `REVERSE` machine-code routine does.

