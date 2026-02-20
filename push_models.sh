#!/usr/bin/env bash
set -euo pipefail

PKG="com.armfinal.translator"
DEST="/sdcard/Android/data/${PKG}/files/models/nllb"
SRC_DIR="${1:-models/nllb_marian}"

adb shell mkdir -p "${DEST}"

if [ ! -d "${SRC_DIR}" ]; then
  echo "Missing source dir: ${SRC_DIR}"
  echo "Expected layout:"
  echo "  ${SRC_DIR}/en_hi/{encoder_model_int8.onnx,decoder_model_int8.onnx,decoder_with_past_model_int8.onnx,source.spm,target.spm,nllb_config.json}"
  echo "  ${SRC_DIR}/hi_en/{encoder_model_int8.onnx,decoder_model_int8.onnx,decoder_with_past_model_int8.onnx,source.spm,target.spm,nllb_config.json}"
  exit 1
fi

adb shell mkdir -p "${DEST}/en_hi" "${DEST}/hi_en"

adb push "${SRC_DIR}/en_hi/encoder_model_int8.onnx" "${DEST}/en_hi/"
adb push "${SRC_DIR}/en_hi/decoder_model_int8.onnx" "${DEST}/en_hi/"
adb push "${SRC_DIR}/en_hi/decoder_with_past_model_int8.onnx" "${DEST}/en_hi/"
adb push "${SRC_DIR}/en_hi/source.spm" "${DEST}/en_hi/"
adb push "${SRC_DIR}/en_hi/target.spm" "${DEST}/en_hi/"
adb push "${SRC_DIR}/en_hi/vocab.json" "${DEST}/en_hi/"
adb push "${SRC_DIR}/en_hi/nllb_config.json" "${DEST}/en_hi/"

adb push "${SRC_DIR}/hi_en/encoder_model_int8.onnx" "${DEST}/hi_en/"
adb push "${SRC_DIR}/hi_en/decoder_model_int8.onnx" "${DEST}/hi_en/"
adb push "${SRC_DIR}/hi_en/decoder_with_past_model_int8.onnx" "${DEST}/hi_en/"
adb push "${SRC_DIR}/hi_en/source.spm" "${DEST}/hi_en/"
adb push "${SRC_DIR}/hi_en/target.spm" "${DEST}/hi_en/"
adb push "${SRC_DIR}/hi_en/vocab.json" "${DEST}/hi_en/"
adb push "${SRC_DIR}/hi_en/nllb_config.json" "${DEST}/hi_en/"

echo "Pushed NLLB models to ${DEST}"
