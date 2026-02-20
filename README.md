# Real-Time On-Device Speech-to-Speech Translator (EN → HI)

> **Bharat AI-SoC Student Challenge — Problem Statement 4**  
> Fully offline English-to-Hindi speech translation on Arm CPU, optimized for Snapdragon 732G

[![Platform](https://img.shields.io/badge/Platform-Android-green?logo=android)](https://developer.android.com)
[![Architecture](https://img.shields.io/badge/Arch-arm64--v8a-blue)](https://developer.arm.com/architectures)
[![API](https://img.shields.io/badge/API-24%2B-brightgreen)](https://developer.android.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Highlights

- **100% offline** — zero cloud dependency, all inference on-device
- **ARM optimized** — NEON SIMD, FP16 vector arithmetic, DOT_PROD instructions
- **Thermal-aware** — 4-tier adaptive pipeline (NORMAL → THROTTLED → EMERGENCY → CRITICAL)
- **Memory-efficient** — peak PSS ~742 MB on a 6 GB device with aggressive lifecycle management
- **Target device** — Xiaomi Redmi Note 10 Pro Max (Snapdragon 732G, Kryo 470)

---

## Architecture

```
English Audio Input (16 kHz Mono PCM)
        │
        ├──► Vosk ASR ──► Live partial transcripts (streaming)
        │
        ├──► Silero VAD ──► Trim silence & split segments
        │         │
        │         ▼
        │    Whisper Tiny ──► Final English transcript
        │         │
        │         ▼
        │    NLLB-200 (INT8+FP16) ──► Hindi translation
        │         │
        │         ▼
        └──► Piper TTS ──► Hindi Audio Output (22.05 kHz)
```

### Five AI Models Working Together

| Stage | Model | Format | Role |
|-------|-------|--------|------|
| **Live ASR** | Vosk Small | Kaldi FST | Streaming partial transcripts while speaking |
| **VAD** | Silero VAD v5 | ONNX | Trim silence, split on pauses >700 ms |
| **Final STT** | Whisper Tiny | GGML (74 MB) | Accurate English transcript after recording |
| **Translation** | NLLB-200-distilled-600M | INT8+FP16 ONNX | English → Hindi neural machine translation |
| **TTS** | Piper | ONNX | Hindi speech synthesis |

---

## ARM Optimizations

Targeting **ARMv8.2-A** instruction set on Snapdragon 732G:

```cmake
-march=armv8.2-a+fp16+dotprod    # ISA target
-O3 -ffast-math                   # Aggressive optimization
```

| Feature | Instruction | Benefit |
|---------|------------|---------|
| **NEON** | 128-bit SIMD | Parallel matrix operations in Whisper & ONNX Runtime |
| **FP16 VA** | Half-precision vectors | 2× throughput for attention computation |
| **DOT_PROD** | `SDOT`/`UDOT` | INT8×INT8→INT32 for quantized NLLB kernels |

**Verified at runtime:**
```
GGML CPU FEATURES: NEON=1 FP16=1 DOTPROD=1
```

### Model Quantization

NLLB-200 is quantized using a hybrid INT8 + FP16 strategy:
- **INT8 dynamic quantization** on MatMul/Gemm operators (per-channel, symmetric)
- **FP16 conversion** for large embedding initializers
- Export script uses subprocess isolation to stay within 8 GB RAM

---

## Thermal Management

The app maps Android `PowerManager` thermal status to four operational modes:

| Mode | Threads | Token Cap | Whisper | NLLB | Fallback |
|------|---------|-----------|---------|------|----------|
| **NORMAL** | 2 | 64 | Yes | Yes | — |
| **THROTTLED** | 1 | 48 | Yes | Yes | — |
| **EMERGENCY** | 1 | 48 | No (Vosk only) | Yes | — |
| **CRITICAL** | 0 | 0 | No | No | Phrase table (300+ entries) |

---

## Memory Lifecycle

Models load lazily and unload aggressively to stay within mobile RAM limits:

| Strategy | Detail |
|----------|--------|
| **Ephemeral Whisper** | Load → infer → free per call (~74 MB reclaimed) |
| **Single Piper voice** | Only one voice loaded at a time |
| **Idle unload** | NLLB after 60s, Piper after 120s |
| **onTrimMemory** | `RUNNING_CRITICAL+`: unload NLLB + Piper; `COMPLETE+`: unload VAD |
| **Direction-specific NLLB** | Only EN→HI models loaded |

**Measured memory (Snapdragon 732G):**

| State | PSS | RSS |
|-------|-----|-----|
| Peak (full pipeline) | ~742 MB | ~848 MB |
| Post-unload (idle) | ~168 MB | ~273 MB |

---

## Performance

Measured on Redmi Note 10 Pro Max (Snapdragon 732G, 6 GB RAM):

| Stage | Time |
|-------|------|
| Silero VAD (2.5s audio) | ~300 ms |
| Whisper Tiny (load + infer) | ~2.2 s |
| NLLB Translation (10 tokens) | ~3.4 s |
| End-to-end pipeline | ~6–8 s |

---

## Project Structure

```
├── app/src/main/
│   ├── java/com/armfinal/translator/
│   │   ├── MainActivity.kt                # Entry point, permissions
│   │   ├── core/
│   │   │   ├── TranslationOrchestrator.kt  # Pipeline coordinator
│   │   │   ├── TranslationViewModel.kt     # UI state management
│   │   │   ├── Models.kt                   # Language, ThermalMode enums
│   │   │   ├── ThermalMonitor.kt           # PowerManager listener
│   │   │   ├── PcmUtils.kt                # Audio utilities
│   │   │   └── MemThermalLogger.kt         # Metrics CSV logger
│   │   ├── asr/VoskEngine.kt               # Streaming ASR
│   │   ├── audio/
│   │   │   ├── AudioRecorder.kt            # 16 kHz PCM capture
│   │   │   └── TtsAudioPlayer.kt           # AudioTrack playback
│   │   ├── nativebridge/NativeBridge.kt    # JNI declarations
│   │   ├── storage/AssetModelManager.kt    # Model extraction
│   │   └── ui/TranslationScreen.kt         # Compose dashboard UI
│   ├── cpp/
│   │   ├── native-lib.cpp                  # 2,871-line native engine
│   │   ├── CMakeLists.txt                  # ARM build flags
│   │   └── third_party/whisper.cpp/        # whisper.cpp source
│   └── assets/models/                      # Bundled models
├── models/
│   ├── export_quantize_nllb.py             # INT8+FP16 quantization script
│   └── download_models.sh                  # Download all models
└── push_models.sh                          # Push NLLB to device via adb
```

---

## Build & Run

### Prerequisites
- Android Studio (Hedgehog+)
- Android SDK 35, NDK, CMake 3.22.1
- arm64-v8a device (Snapdragon 732G recommended)

### Steps

```bash
# 1. Clone
git clone https://github.com/rohan-js/arm-speech-translator.git
cd arm-speech-translator

# 2. Download models
cd models && bash download_models.sh && cd ..

# 3. Export & quantize NLLB (requires Python 3.10+)
cd models && python export_quantize_nllb.py && cd ..

# 4. Push NLLB models to device
bash push_models.sh models/nllb_marian

# 5. Open in Android Studio → Build → Run on arm64 device
```

### Model Placement

| Model | Location | Size |
|-------|----------|------|
| Silero VAD | `assets/models/silero/` | ~2 MB |
| Whisper Tiny | `assets/models/whisper/` | ~74 MB |
| Vosk EN/HI | `assets/models/vosk/` | ~50 MB each |
| Piper voices | `assets/models/piper/` | ~60 MB each |
| NLLB (INT8) | External via `push_models.sh` | ~400 MB per direction |
| Phrase tables | `assets/models/fallback/` | ~20 KB |

---

## Testing

```bash
# Instrumentation smoke test
./gradlew connectedAndroidTest
```

**Built-in debug tools** (in the app UI):
- **VAD Test (5s)** — Record → run VAD → log segment boundaries
- **ASR Test (5s)** — Record → VAD + Whisper → log transcriptions
- **Thermal Override** — Manually trigger any thermal mode
- **Runtime Stats** — Live model load status, thread counts, token counts

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| App Language | Kotlin 1.9.24 |
| Native Engine | C++17 (NDK) |
| UI | Jetpack Compose 1.6.8 (Material 3) |
| STT (Streaming) | Vosk 0.3.47 |
| STT (Final) | whisper.cpp (GGML) |
| VAD | Silero VAD v5 (ONNX) |
| Translation | NLLB-200-distilled-600M (INT8 ONNX) |
| TTS | Piper (ONNX) |
| Inference | ONNX Runtime Mobile (CPU) |
| Target ISA | ARMv8.2-A (NEON + FP16VA + DOTPROD) |

---

*Built for the Bharat AI-SoC Student Challenge — Problem Statement 4*  
*Real-Time On-Device Speech-to-Speech Translation using NEON on Arm CPU*
