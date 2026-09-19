"""Environment captured alongside every result.

Results are compared across GPUs, so a result that does not record which GPU
produced it cannot be placed in that comparison afterwards. Every run records
its own hardware rather than relying on the filename or on notes kept
elsewhere.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone


def run_metadata(args=None) -> dict:
    """Hardware, software and invocation for the run now starting."""
    meta = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv,
    }
    if args is not None:
        meta["args"] = {k: str(v) for k, v in vars(args).items()}

    try:
        import torch
        meta["torch"] = torch.__version__
        meta["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            meta["gpu"] = props.name
            meta["gpu_total_gb"] = round(props.total_memory / (1024 ** 3), 2)
    except Exception as e:
        meta["torch_error"] = f"{type(e).__name__}: {e}"

    for field in ("driver_version", "pcie.link.gen.max", "pcie.link.width.max"):
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15,
            )
            if out.returncode == 0 and out.stdout.strip():
                meta[field.replace(".", "_")] = out.stdout.strip().splitlines()[0].strip()
        except Exception:
            pass

    return meta


if __name__ == "__main__":
    import json
    print(json.dumps(run_metadata(), indent=2))
