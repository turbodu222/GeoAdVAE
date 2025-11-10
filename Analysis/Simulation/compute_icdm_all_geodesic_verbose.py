
"""
compute_icdm_all_geodesic_verbose (robust & verbose replacement)

- Drop-in compatible with cajal.sample_swc.compute_icdm_all_geodesic
- Adds retries with tiny perturbations per file
- Writes header FIRST (guaranteed), supports header name 'cell_id' or 'cellid'
- Prints detailed summary (#files, #ok, #failed)
"""

import os
import csv
import glob
import itertools as it
import numpy as np
from typing import Optional

import cajal.sample_swc as cs
import cajal.swc as swc


def _write_header(writer: csv.writer, n: int, header_name: str) -> None:
    assert header_name in ("cell_id", "cellid")
    header = [header_name] + [f"d_{i}_{j}" for i, j in it.combinations(range(n), 2)]
    writer.writerow(header)


def _jitter_and_rescale_swc_file(src_path: str, dst_path: str,
                                 scale_eps: float = 1e-2,
                                 coord_eps: float = 1e-6) -> None:
    sx = float(1.0 + np.random.uniform(-scale_eps, scale_eps))
    sy = float(1.0 + np.random.uniform(-scale_eps, scale_eps))
    sz = float(1.0 + np.random.uniform(-scale_eps, scale_eps))
    with open(src_path, "r") as fin, open(dst_path, "w") as fout:
        for line in fin:
            if not line.strip() or line.lstrip().startswith("#"):
                fout.write(line); continue
            parts = line.strip().split()
            if len(parts) < 7:
                fout.write(line); continue
            try:
                x = float(parts[2]); y = float(parts[3]); z = float(parts[4])
                x = x * sx + np.random.uniform(-coord_eps, coord_eps)
                y = y * sy + np.random.uniform(-coord_eps, coord_eps)
                z = z * sz + np.random.uniform(-coord_eps, coord_eps)
                parts[2] = f"{x:.9g}"; parts[3] = f"{y:.9g}"; parts[4] = f"{z:.9g}"
                fout.write(" ".join(parts) + "\n")
            except Exception:
                fout.write(line)


def _geodesic_icdm_with_retries(file_path: str,
                                n_sample: int,
                                preprocess,
                                max_retries: int = 10,
                                scale_eps: float = 1e-2,
                                coord_eps: float = 1e-6,
                                tmp_dir: Optional[str] = None):
    if tmp_dir is None:
        tmp_dir = os.path.join(os.path.dirname(file_path), "_tmp_geodesic_safe")
    os.makedirs(tmp_dir, exist_ok=True)

    last_path = file_path
    for attempt in range(max_retries + 1):
        try:
            forest, _ = swc.read_swc(last_path)
            tree = preprocess(forest)
            if isinstance(tree, swc.Err):
                return None
            dist_vec, _node_types = cs.icdm_geodesic(tree, n_sample)
            return dist_vec
        except Exception:
            if attempt == max_retries:
                return None
            dst_path = os.path.join(tmp_dir, f"jitter_{os.path.basename(file_path)}.{attempt+1}.swc")
            _jitter_and_rescale_swc_file(last_path, dst_path, scale_eps=scale_eps, coord_eps=coord_eps)
            last_path = dst_path
    return None


def compute_icdm_all_geodesic(infolder: str,
                              out_csv: str,
                              n_sample: int,
                              num_processes: int = 8,
                              preprocess=(lambda forest: forest[0]),
                              # extra options
                              max_retries: int = 10,
                              scale_eps: float = 1e-2,
                              coord_eps: float = 1e-6,
                              header_name: str = "cellid") -> int:
    """
    Args are the same as CAJAL's, plus:
      - max_retries / scale_eps / coord_eps: robustness controls
      - header_name: choose 'cell_id' (newer) or 'cellid' (older) for compatibility with run_gw

    Returns:
      int: number of successfully processed cells (rows written)
    """
    swc_files = sorted(glob.glob(os.path.join(infolder, "*.swc")))
    total = len(swc_files)
    if total == 0:
        raise RuntimeError(f"No .swc files found in {infolder}")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    ok = 0; failed = []

    # Always write header first
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        _write_header(writer, n_sample, header_name)

        for i, fp in enumerate(swc_files, 1):
            cell_id = os.path.basename(fp)
            dist_vec = _geodesic_icdm_with_retries(
                file_path=fp,
                n_sample=n_sample,
                preprocess=preprocess,
                max_retries=max_retries,
                scale_eps=scale_eps,
                coord_eps=coord_eps
            )
            if dist_vec is None:
                failed.append(cell_id); continue
            if dist_vec.shape[0] != (n_sample * (n_sample - 1)) // 2:
                failed.append(cell_id); continue

            writer.writerow([cell_id] + dist_vec.tolist())
            ok += 1

    print(f"compute_icdm_all_geodesic: total={total}, ok={ok}, failed={len(failed)}")
    if failed:
        print("  failed list (first 50):", failed[:50])
    return ok
