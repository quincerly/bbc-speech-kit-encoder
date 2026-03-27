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
Speech System fitted. It chains a BBC BASIC program (`PLAYER`) which
streams the LPC data to the chip via `SOUND &FF60` / `SOUND &FF00`.

