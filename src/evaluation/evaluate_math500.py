#!/usr/bin/env python3
"""
evaluate_math500.py
───────────────────
Evaluate baseline + AE‑compressed KV‑cache models on the 500‑problem
MATH subset, driven by your standard JSON config.
"""

import argparse, json, importlib.util
from pathlib import Path

# ── 1  register pure‑Python task (no YAML) ─────────────────────────────────────
def ensure_math500_task():
    import lm_eval
    task_dir = Path(lm_eval.__file__).parent / "tasks" / "math500"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "__init__.py").write_text(
        "from lm_eval.tasks.hendrycks_math import HendrycksMath\n"
        "class Math500(HendrycksMath):\n"
        "    DATASET_PATH = 'HuggingFaceH4/MATH-500'\n"
    )
    print("✓ math500 Python task refreshed")

# ── 2  dynamic wrapper that compresses the KV cache ────────────────────────────
def build_wrapper(model_name, latent_dim, bits, ae_ckpt):
    from lm_eval.models.huggingface import HFCausalLM
    spec = importlib.util.spec_from_file_location("benchmark", "./benchmark.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

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

# ── 3  main driver ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))

    model_name  = cfg["model_name"]
    ae_ckpt     = cfg["autoencoder_path"]
    latent_dim  = cfg["latent_dim"]
    bits_list   = cfg["quantization_bits"]
    out_dir     = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    batch_size  = cfg.get("batch_size", 4)
    device      = cfg.get("device", "cuda")

    ensure_math500_task()
    from lm_eval import evaluator

    # baseline
    print(f"\n▶ Baseline : {model_name}")
    baseline = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_name}",
        tasks=["math500"],
        batch_size=batch_size,
        device=device,
    )
    base_acc = baseline["results"]["math500"]["exact_match"]
    print(f"   exact‑match {base_acc:.3f}")
    json.dump(baseline, open(out_dir/"math500_baseline.json","w"), indent=2)

    # compressed runs
    for bits in bits_list:
        tag = f"lat{latent_dim}_bits{bits}"
        print(f"\n▶ Compressed : latent={latent_dim}  bits={bits}")
        Wrapper = build_wrapper(model_name, latent_dim, bits, ae_ckpt)
        res = evaluator.simple_evaluate(
            model=Wrapper,
            model_args="",
            tasks=["math500"],
            batch_size=batch_size,
            device=device,
        )
        acc = res["results"]["math500"]["exact_match"]
        ratio = (64/latent_dim)*(16/bits)
        print(f"   accuracy {acc:.3f}   (compression ×{ratio:.1f})")
        json.dump(res, open(out_dir/f"math500_{tag}.json","w"), indent=2)

if __name__ == "__main__":
    main()