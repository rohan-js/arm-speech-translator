# Model Layout

Place runtime model artifacts under this directory before packaging a demo APK.

Expected layout:

- `silero/silero_vad.onnx`
- `vosk/en/...`
- `vosk/hi/...`
- `whisper/ggml-tiny.bin`
- `nllb/encoder_model_int8.onnx`
- `nllb/decoder_model_int8.onnx`
- `nllb/decoder_with_past_model_int8.onnx`
- `nllb/tokenizer.model`
- `sanity/mul_1.onnx`
- `piper/en_US-lessac-medium.onnx`
- `piper/en_US-lessac-medium.onnx.json`
- `piper/hi_IN-pratham-medium.onnx`
- `piper/hi_IN-pratham-medium.onnx.json`
- `espeak-ng-data/...`
- `fallback/phrases_en_hi.tsv`
- `fallback/phrases_hi_en.tsv`
