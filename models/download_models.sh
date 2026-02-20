#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${ROOT_DIR}/artifacts"
mkdir -p "${OUT_DIR}"

# Silero VAD
mkdir -p "${OUT_DIR}/silero"
curl -L "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx" \
  -o "${OUT_DIR}/silero/silero_vad.onnx"

# Vosk EN small
mkdir -p "${OUT_DIR}/vosk"
curl -L "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" \
  -o "${OUT_DIR}/vosk/en.zip"

# Vosk HI small (update URL if upstream changes)
curl -L "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip" \
  -o "${OUT_DIR}/vosk/hi.zip"

# Whisper tiny model for whisper.cpp
mkdir -p "${OUT_DIR}/whisper"
curl -L "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin" \
  -o "${OUT_DIR}/whisper/ggml-tiny.bin"

# Piper voices (EN + HI)
mkdir -p "${OUT_DIR}/piper"
curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx?download=true" \
  -o "${OUT_DIR}/piper/en_US-lessac-medium.onnx"
curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json?download=true" \
  -o "${OUT_DIR}/piper/en_US-lessac-medium.onnx.json"
curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx?download=true" \
  -o "${OUT_DIR}/piper/hi_IN-pratham-medium.onnx"
curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx.json?download=true" \
  -o "${OUT_DIR}/piper/hi_IN-pratham-medium.onnx.json"

echo "Downloads complete."
