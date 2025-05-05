#!/usr/bin/env python3
"""
evaluate_math500.py   –   run baseline + AE‑compressed KV on MATH‑500
using the same JSON config you already use for WikiText.

Works with the newest lm‑evaluation‑harness (no YAML needed).
"""

import argparse, json, importlib.util
from pathlib import Path

# ── 1  inline task registration (no YAML, no forbidden keys) ──────────────────
def register_math500_inline():
    from lm_eval.tasks import TASK_REGISTRY
    try:
        # Attempt direct import (preferred method)
        from lm_eval.tasks.hendrycks_math import HendrycksMath
    except ImportError:
        # Fallback: dynamically retrieve the task if import fails
        from lm_eval.tasks import get_task
        try:
            HendrycksMath = get_task("hendrycks_math").__class__
            print("Warning: Direct import failed, using fallback via get_task('hendrycks_math')")
        except KeyError:
            raise ImportError(
                "Cannot find 'HendrycksMath'. Ensure lm_eval is up-to-date and includes the MATH task. "
                "Try running 'pip install --upgrade lm-eval'."
            )

    class Math500Inline(HendrycksMath):
        """500‑problem slice of Hendrycks MATH (HuggingFaceH4/MATH‑500)."""
        DATASET_PATH = "HuggingFaceH4/MATH-500"
        DATASET_NAME = None

    TASK_REGISTRY["math500_inline"] = Math500Inline
    print("✓ inline task 'math500_inline' registered")

# ── 2  dynamic wrapper that compresses the KV cache each forward pass ─────────
def build_wrapper(model_name, latent_dim, bits, ae_ckpt):
    from lm_eval.models.huggingface import HFCausalLM
    spec = importlib.util.spec_from_file_location("benchmark", "./benchmark.py")
    bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)

    class HFAECompressed(HFCausalLM):
        def __init__(self, **kw):
            super().__init__(pretrained=model_name, **kw)
            self.aes = bench.build_autoencoders(self.model, latent_dim, ae_ckpt)
            self.bits = bits
        def _model_call(self, inputs, past_key_values=None, **kw):
            out = self.model(input_ids=inputs,
                             past_key_values=past_key_values,
                             use_cache=True, **kw)
            out.past_key_values = bench.compress_past(out.past_key_values,
                                                      self.aes, self.bits)
            return out
    return HFAECompressed

# ── 3  main driver ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.load(open(args.config))

    model_name   = cfg["model_name"]
    ae_ckpt      = cfg["autoencoder_path"]
    latent_dim   = cfg["latent_dim"]
    bits_list    = cfg["quantization_bits"]
    out_dir      = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    batch_size   = cfg.get("batch_size", 4)
    device       = cfg.get("device", "cuda")

    # Register inline task before importing evaluator
    register_math500_inline()
    from lm_eval import evaluator

    # ── baseline
    print(f"\n▶ Baseline : {model_name}")
    baseline = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_name}",
        tasks=["math500_inline"],
        batch_size=batch_size,
        device=device,
    )
    base_acc = baseline["results"]["math500_inline"]["exact_match"]
    print(f"   exact‑match {base_acc:.3f}")
    json.dump(baseline, open(out_dir/"math500_baseline.json","w"), indent=2)

    # ── compressed runs
    for bits in bits_list:
        tag = f"lat{latent_dim}_bits{bits}"
        print(f"\n▶ Compressed : latent={latent_dim}  bits={bits}")
        Wrapper = build_wrapper(model_name, latent_dim, bits, ae_ckpt)
        res = evaluator.simple_evaluate(
            model=Wrapper,
            model_args="",
            tasks=["math500_inline"],
            batch_size=batch_size,
            device=device,
        )
        acc = res["results"]["math500_inline"]["exact_match"]
        ratio = (64/latent_dim)*(16/bits)
        print(f"   accuracy {acc:.3f}   (compression ×{ratio:.1f})")
        json.dump(res, open(out_dir/f"math500_{tag}.json","w"), indent=2)

if __name__ == "__main__":
    main()