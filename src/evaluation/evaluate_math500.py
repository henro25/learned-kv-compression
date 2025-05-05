#!/usr/bin/env python3
"""
evaluate_math500.py  ─ config‑driven kv‑compression accuracy on MATH‑500
"""

import argparse, json, importlib.util
from pathlib import Path

# ── lm‑eval helpers ────────────────────────────────────────────────────────────
def ensure_math500_registered():
    import lm_eval, textwrap
    task_dir = Path(lm_eval.__file__).parent / "tasks" / "math500"
    if task_dir.exists():
        return
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.yaml").write_text(textwrap.dedent("""
        group: math
        task: math500
        dataset_path: HuggingFaceH4/MATH-500
        output_type: generation
        doc_to_text: "Problem: {{problem}}\\nAnswer:"
        doc_to_target: "{{answer}}"
        metric_list:
          - metric: exact_match
    """))
    (task_dir / "__init__.py").write_text(
        "from lm_eval.tasks.hendrycks_math import HendrycksMath\n"
        "class Math500(HendrycksMath):\n"
        "    DATASET_PATH = 'HuggingFaceH4/MATH-500'\n")
    print("✓ Registered math500 task")

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

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--config", required=True)
    cfg = json.load(open(argp.parse_args().config))

    model_name = cfg["model_name"]
    ae_ckpt    = cfg["autoencoder_path"]
    latent     = cfg["latent_dim"]
    bits_list  = cfg["quantization_bits"]
    out_dir    = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    device     = cfg.get("device", "cuda")
    batch_size = cfg.get("batch_size", 4)

    ensure_math500_registered()
    from lm_eval import evaluator

    # baseline with official HuggingFace backend "hf"
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
        tag = f"lat{latent}_bits{bits}"
        print(f"\n▶ Compressed : latent={latent}  bits={bits}")
        Wrapped = build_wrapper(model_name, latent, bits, ae_ckpt)
        res = evaluator.simple_evaluate(
            model=Wrapped,
            model_args="",
            tasks=["math500"],
            batch_size=batch_size,
            device=device,
        )
        acc = res["results"]["math500"]["exact_match"]
        ratio = (64/latent)*(16/bits)
        print(f"   accuracy {acc:.3f}   (compression ×{ratio:.1f})")
        json.dump(res, open(out_dir/f"math500_{tag}.json","w"), indent=2)

if __name__ == "__main__":
    main()