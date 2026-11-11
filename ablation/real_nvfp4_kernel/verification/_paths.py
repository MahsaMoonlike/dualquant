"""sys.path setup shared by the verification scripts: repo root, legacy_vendor, this kernel dir."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.dirname(_HERE)
ROOT = os.path.dirname(os.path.dirname(KDIR))
for p in (ROOT, os.path.join(ROOT, "legacy_vendor"), KDIR):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("BLOCKFMT_BACKEND", "nvfp4_numerics")
MODEL = os.environ.get("DQ_MODEL", "unsloth/Meta-Llama-3.1-8B")


def load_weight(name):
    """One tensor straight from the safetensors shards (no full model load)."""
    import json
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    d = snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors"])
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(d, idx[name]), "pt") as f:
        return f.get_tensor(name)
