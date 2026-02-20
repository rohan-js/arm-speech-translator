#!/usr/bin/env python3
"""
Export and quantize facebook/nllb-200-distilled-600M to ONNX for mobile.

DESIGNED FOR 8 GB RAM MACHINES:
  Every heavy step (export, quantize, fp16-convert) runs in its OWN subprocess.
  When the subprocess exits, the OS reclaims ALL its memory — guaranteed.
  Peak RAM per step:
    - Optimum export:    ~4 GB (subprocess 1)
    - int8 quantize:     ~2× model file (subprocess per model)
    - fp16 post-process: ~2× model file (subprocess per model)
  Steps NEVER overlap.

Strategy (hybrid int8 + fp16):
  1. Export to ONNX fp32 via optimum-cli (subprocess)
  2. quantize_dynamic on MatMul/Gemm → int8 weights (subprocess per model)
  3. Convert remaining large fp32 initializers to fp16 (subprocess per model)
     — this catches the ~1 GB embedding table that quantize_dynamic skips

Usage:
  pip install optimum[onnxruntime] onnxruntime onnx numpy
  python export_quantize_nllb.py --out ./artifacts/nllb

Then copy the 5 output files to:
  app/src/main/assets/models/nllb/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


PYTHON = sys.executable
MODEL_ID = "facebook/nllb-200-distilled-600M"
MODEL_NAMES = ["encoder_model", "decoder_model", "decoder_with_past_model"]


def run_isolated(label: str, code: str, timeout: int = 1800) -> None:
    """Run *code* in a fresh subprocess. All RAM is freed on exit."""
    print(f"\n  → Launching subprocess: {label}")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, "-c", code],
        env=env,
        timeout=timeout,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise SystemExit(f"Subprocess '{label}' failed (exit {result.returncode})")
    print(f"  ✓ {label} finished in {elapsed:.0f}s")


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export + int8+fp16 quantize NLLB for mobile (8 GB RAM safe)"
    )
    parser.add_argument("--out", required=True, type=Path,
                        help="Output directory (should be on project drive)")
    parser.add_argument("--keep-fp32", action="store_true",
                        help="Keep intermediate fp32 ONNX export (~2.4 GB)")
    parser.add_argument("--skip-export", action="store_true",
                        help="Reuse existing fp32 in <out>/onnx_fp32")
    args = parser.parse_args()

    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    export_dir = out / "onnx_fp32"
    quant_dir = out / "onnx_int8"
    quant_dir.mkdir(exist_ok=True)

    # ── Step 1: Export fp32 via optimum-cli (subprocess) ─────────────────
    if args.skip_export and export_dir.exists():
        print(f"[1/5] Skipping export — reusing {export_dir}")
    else:
        print(f"[1/5] Exporting {MODEL_ID} to ONNX fp32 (in subprocess) ...")
        export_code = f'''
import gc, os
os.environ["OMP_NUM_THREADS"] = "1"
from optimum.onnxruntime import ORTModelForSeq2SeqLM
print("  Loading model from HuggingFace ...")
model = ORTModelForSeq2SeqLM.from_pretrained("{MODEL_ID}", export=True)
print("  Saving fp32 ONNX ...")
model.save_pretrained("{export_dir}")
del model
gc.collect()
print("  Export done.")
'''
        run_isolated("optimum-export", export_code, timeout=3600)

    # Verify fp32 files exist
    for name in MODEL_NAMES:
        fp32 = export_dir / f"{name}.onnx"
        if not fp32.exists():
            raise SystemExit(f"Expected {fp32} but not found. Export may have failed.")
        print(f"  fp32: {name}.onnx = {fp32.stat().st_size / (1024**2):.0f} MB")

    # ── Step 2: int8 quantize each model (one subprocess per model) ──────
    print(f"\n[2/5] Quantizing linear layers to int8 (one model at a time) ...")
    for name in MODEL_NAMES:
        src = export_dir / f"{name}.onnx"
        dst = quant_dir / f"{name}_int8.onnx"

        if dst.exists():
            print(f"  ↩ {dst.name} already exists ({dst.stat().st_size/(1024**2):.0f} MB), skipping")
            continue

        quant_code = f'''
import time, os
os.environ["OMP_NUM_THREADS"] = "1"
from onnxruntime.quantization import QuantType, quantize_dynamic
src = "{src}"
dst = "{dst}"
size_mb = os.path.getsize(src) / (1024**2)
print(f"  Quantizing {{src}} ({{size_mb:.0f}} MB) ...")
t0 = time.time()
quantize_dynamic(
    model_input=src,
    model_output=dst,
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=True,
    extra_options={{"WeightSymmetric": True, "ActivationSymmetric": False}},
    op_types_to_quantize=["MatMul", "Gemm"],
)
out_mb = os.path.getsize(dst) / (1024**2)
print(f"  int8 result: {{out_mb:.0f}} MB ({{out_mb/size_mb*100:.0f}}% of fp32) in {{time.time()-t0:.0f}}s")
'''
        run_isolated(f"int8-{name}", quant_code)

    # ── Step 3: fp16 post-process remaining large fp32 initializers ──────
    print(f"\n[3/5] Converting remaining large fp32 tensors to fp16 ...")
    for name in MODEL_NAMES:
        dst = quant_dir / f"{name}_int8.onnx"
        if not dst.exists():
            continue

        before_mb = dst.stat().st_size / (1024**2)
        fp16_code = f'''
import os, time, gc
os.environ["OMP_NUM_THREADS"] = "1"
import onnx
import numpy as np
from onnx import numpy_helper, TensorProto

model_path = "{dst}"
print(f"  Loading {{model_path}} ...")
t0 = time.time()
model = onnx.load(model_path, load_external_data=True)

converted = 0
saved_bytes = 0
for init in model.graph.initializer:
    if init.data_type != TensorProto.FLOAT:
        continue
    raw_size = len(init.raw_data) if init.raw_data else len(init.float_data) * 4
    if raw_size < 4096:
        continue
    arr = numpy_helper.to_array(init).astype(np.float16)
    new_t = numpy_helper.from_array(arr, name=init.name)
    init.CopyFrom(new_t)
    saved_bytes += raw_size - len(init.raw_data)
    converted += 1
    del arr, new_t
    if converted % 20 == 0:
        gc.collect()

# Update graph input types for converted initializers
init_names = {{i.name for i in model.graph.initializer}}
for inp in model.graph.input:
    if inp.name in init_names:
        tt = inp.type.tensor_type
        if tt.elem_type == TensorProto.FLOAT:
            for i2 in model.graph.initializer:
                if i2.name == inp.name and i2.data_type == TensorProto.FLOAT16:
                    tt.elem_type = TensorProto.FLOAT16
                    break

print(f"  Converted {{converted}} tensors, saved ~{{saved_bytes/(1024**2):.0f}} MB")
print(f"  Saving ...")
onnx.save(model, model_path)
del model
gc.collect()
print(f"  Done in {{time.time()-t0:.0f}}s")
'''
        run_isolated(f"fp16-{name}", fp16_code)
        after_mb = dst.stat().st_size / (1024**2)
        print(f"  {name}_int8.onnx: {before_mb:.0f} → {after_mb:.0f} MB")

    # ── Step 4: Copy tokenizer + config ──────────────────────────────────
    print(f"\n[4/5] Copying tokenizer and generating config ...")
    tokenizer_dst = quant_dir / "tokenizer.model"
    for candidate in [
        export_dir / "sentencepiece.bpe.model",
        export_dir / "tokenizer.model",
        export_dir / "source.spm",
    ]:
        if candidate.exists():
            shutil.copy2(candidate, tokenizer_dst)
            print(f"  ✓ tokenizer.model from {candidate.name} "
                  f"({tokenizer_dst.stat().st_size / 1024:.0f} KB)")
            break
    else:
        print("  ⚠ tokenizer.model not found — copy manually")

    config = {
        "eos_id": 2, "pad_id": 1, "unk_id": 3,
        "eng_Latn": 256047, "hin_Deva": 256068,
    }
    (quant_dir / "nllb_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"  ✓ nllb_config.json")

    # ── Step 5: Cleanup + summary ────────────────────────────────────────
    if not args.keep_fp32:
        print(f"\nCleaning up fp32 export: {export_dir}")
        shutil.rmtree(export_dir, ignore_errors=True)

    print(f"\n[5/5] Final output files:")
    total_mb = 0.0
    for f in sorted(quant_dir.iterdir()):
        if f.is_file():
            mb = f.stat().st_size / (1024**2)
            total_mb += mb
            print(f"  {f.name:48s} {mb:8.1f} MB")
    print(f"  {'TOTAL':48s} {total_mb:8.1f} MB")

    if total_mb > 800:
        print(f"\n⚠ Total {total_mb:.0f} MB exceeds 800 MB target — check fp16 step output.")
    else:
        print(f"\n✅ Models look well-sized for mobile deployment.")

    print(f"\nCopy these files to app/src/main/assets/models/nllb/ and rebuild.")


if __name__ == "__main__":
    main()
