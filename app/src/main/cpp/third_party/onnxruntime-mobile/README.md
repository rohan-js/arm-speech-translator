# ONNX Runtime prebuilt package

Place the prebuilt ONNX Runtime mobile C/C++ package here.

Expected layout for `arm64-v8a` static linking:

- `include/onnxruntime_cxx_api.h`
- `lib/arm64-v8a/libonnxruntime*.a` or `lib/libonnxruntime*.a`

This project links ONNX Runtime from `app/src/main/cpp/CMakeLists.txt` using:

- `ONNXRUNTIME_MOBILE_ROOT` (CMake cache variable)

If you keep the package in this default folder, no extra config is required.
