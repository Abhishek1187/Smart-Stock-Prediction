import argparse
import json
import os
import shutil
import subprocess

import tensorflow as tf


def find_nvidia_smi():
    in_path = shutil.which("nvidia-smi")
    if in_path:
        return in_path

    candidate_paths = [
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        r"C:\Windows\System32\nvidia-smi.exe",
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    return None


def query_nvidia_smi():
    exe = find_nvidia_smi()
    if not exe:
        return {"available": False, "reason": "nvidia_smi_not_found", "gpus": []}

    cmd = [
        exe,
        "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {
            "available": False,
            "reason": f"nvidia_smi_error:{result.stderr.strip()}",
            "gpus": [],
            "path": exe,
        }

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        gpus.append(
            {
                "name": parts[0],
                "driver_version": parts[1],
                "memory_total_mb": float(parts[2]),
                "memory_used_mb": float(parts[3]),
                "utilization_gpu_percent": float(parts[4]),
                "temperature_c": float(parts[5]),
            }
        )

    return {"available": True, "reason": "ok", "gpus": gpus, "path": exe}


def main():
    parser = argparse.ArgumentParser(description="GPU/CUDA preflight for TensorFlow training")
    parser.add_argument("--require-gpu", action="store_true", help="Exit non-zero when GPU is not detected")
    args = parser.parse_args()

    tf_gpus = tf.config.list_physical_devices("GPU")
    for gpu in tf_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    status = {
        "tf_version": tf.__version__,
        "tf_built_with_cuda": bool(tf.test.is_built_with_cuda()),
        "tf_visible_gpu_count": int(len(tf_gpus)),
        "tf_visible_gpus": [g.name for g in tf_gpus],
        "nvidia_smi": query_nvidia_smi(),
    }

    print(json.dumps(status, indent=2))

    if args.require_gpu and status["tf_visible_gpu_count"] == 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
