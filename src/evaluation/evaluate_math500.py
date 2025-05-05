#!/usr/bin/env python3
"""
evaluate_math500.py   (config‑driven version)

Run exact‑match accuracy on the MATH‑500 dataset for
  • baseline HuggingFace model
  • AE‑compressed KV‑cache for every quantisation bit listed in the config

Config JSON example (same schema you already use):

{
  "model_name": "distilgpt2",
  "autoencoder_path": "experiment_results_distilgpt2/.../autoencoders_final.pth",
  "latent_dim": 8,
  "quantization_bits": [2, 4, 8, 16],
  "output_dir": "experiment_results_distilgpt2",
  "device": "cuda",
  "batch_size": 4
}
"""

import argparse, json, os, statistics, importlib.util, types
from pathlib import Path

# ----------------------------- lm‑eval helpers ---------------------------------
def ensure_math500_registered():
    """Add a 'math500' task to lm_eval if not present."""
    import lm_eval
    task_dir = Path(lm_eval.__file__).parent / "tasks" / "math500"
    if task_dir.exists():
        return
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.yaml").write_text(
        "group: math\n"
        "task: math500\n"
        "dataset_path: HuggingFaceH4/MATH-500\n"
        "output_type: generation\n"
        "doc_to_text: \"Problem: {{problem}}\\nAnswer:\"\n"
        "doc_to_target: \"{{answer}}\"\n"
        "metric_list:\n  - metric: exact_match\n")
    (task_dir / "__init__.py").write_text(
        "from lm_eval.tasks.hendrycks_math import HendrycksMath\n"
        "class Math500(HendrycksMath):\n"
        "    DATASET_PATH = 'HuggingFaceH4/MATH-500'\n")
    print("✓ Registered math500 task in lm_eval")

def build_compressed_wrapper(model_name:str, latent_dim:int, bits:int, ae_ckpt:str):
    """
    Return a dynamic subclass of HFCausalLM that compresses past_key_values
    at every step using your existing benchmark helpers.
    """
    from lm_eval.models.huggingface import HFCausalLM
    # load user's benchmark.py utilities dynamically
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

# ---------------------------------- main --------------------------------------
def main():
    argp = argparse.ArgumentParser(description="Evaluate baseline vs. AE‑compressed on MATH‑500")
    argp.add_argument("--config", required=True, help="Path to JSON config file")
    cfg = json.load(open(argp.parse_args().config))

    model_name      = cfg["model_name"]
    ae_ckpt         = cfg["autoencoder_path"]
    latent_dim      = cfg["latent_dim"]
    bits_list       = cfg["quantization_bits"]
    output_dir      = Path(cfg["output_dir"])
    device          = cfg.get("device", "cuda")
    batch_size      = cfg.get("batch_size", 4)

    ensure_math500_registered()
    from lm_eval import evaluator

    # ------------ baseline -----------------------------------------------------
    print("\n▶ Running baseline :", model_name)
    baseline_res = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=f"pretrained={model_name}",
        tasks=["math500"],
        batch_size=batch_size,
        device=device,
    )
    base_acc = baseline_res["results"]["math500"]["exact_match"]
    print(f"   exact‑match = {base_acc:.3f}")
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(baseline_res, open(output_dir/"math500_baseline.json","w"), indent=2)

    # ------------ compressed loops -------------------------------------------
    for bits in bits_list:
        tag = f"lat{latent_dim}_bits{bits}"
        print(f"\n▶ Compressed run : latent={latent_dim}  bits={bits}")
        Wrapped = build_compressed_wrapper(model_name, latent_dim, bits, ae_ckpt)
        res = evaluator.simple_evaluate(
            model=Wrapped,
            model_args="",
            tasks=["math500"],
            batch_size=batch_size,
            device=device,
        )
        acc = res["results"]["math500"]["exact_match"]
        ratio = (64/latent_dim)*(16/bits)
        print(f"   accuracy {acc:.3f}   (compression ×{ratio:.1f})")
        json.dump(res, open(output_dir/f"math500_{tag}.json","w"), indent=2)

if __name__ == "__main__":
    main()