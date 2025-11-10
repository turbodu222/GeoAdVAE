# run_geodesic_icdm_safe.py
import os, sys, io, csv, glob, shutil, itertools as it
import numpy as np

import cajal.sample_swc as cs
import cajal.swc as swc

def jitter_and_rescale_swc_file(src_path, dst_path, scale_eps=1e-2, coord_eps=1e-6):
    """
    产生一个副本：对非注释行的 x,y,z 做极小各向异性缩放(±scale_eps)并加微小抖动(±coord_eps)。
    不改变行数/列数/父子关系/半径/类型等。
    """
    # 随机各向异性缩放因子
    sx = float(1.0 + np.random.uniform(-scale_eps, scale_eps))
    sy = float(1.0 + np.random.uniform(-scale_eps, scale_eps))
    sz = float(1.0 + np.random.uniform(-scale_eps, scale_eps))

    with open(src_path, "r") as fin, open(dst_path, "w") as fout:
        for line in fin:
            if not line.strip() or line.lstrip().startswith("#"):
                fout.write(line)
                continue
            parts = line.strip().split()
            if len(parts) < 7:
                # 保守写回
                fout.write(line)
                continue
            try:
                # id type x y z radius parent
                x = float(parts[2]); y = float(parts[3]); z = float(parts[4])
                # 对坐标做轻微各向异性缩放与极小抖动
                x = x * sx + np.random.uniform(-coord_eps, coord_eps)
                y = y * sy + np.random.uniform(-coord_eps, coord_eps)
                z = z * sz + np.random.uniform(-coord_eps, coord_eps)
                parts[2] = f"{x:.9g}"
                parts[3] = f"{y:.9g}"
                parts[4] = f"{z:.9g}"
                fout.write(" ".join(parts) + "\n")
            except Exception:
                # 解析失败则原样写回
                fout.write(line)

def write_header(writer, n):
    # 列名与 CAJAL 的 write_csv_block 一致的风格
    header = ["cell_id"] + [f"d_{i}_{j}" for i, j in it.combinations(range(n), 2)]
    writer.writerow(header)

def geodesic_icdm_for_file(
    file_path,
    n_sample=100,
    types=(1,3,4),
    max_retries=10,
    scale_eps=1e-2,
    coord_eps=1e-6,
    tmp_dir=None
):
    """
    对单个 SWC 文件尝试计算 geodesic ICDM。
    若二分超时，生成轻微扰动的临时副本并重试（不改原文件）。
    成功返回 (dist_vec, node_types)；失败返回 None。
    """
    # 基础预处理（与你当前调用一致）
    preprocess = swc.preprocessor_geo(list(types))

    # 尝试：原文件 + 若干副本
    tried_paths = [file_path]
    cleanup_paths = []
    try_id = 0
    last_exc = None

    while try_id < (1 + max_retries):
        this_path = tried_paths[try_id]

        try:
            tree = preprocess(swc.read_swc(this_path)[0])
            if isinstance(tree, swc.Err):
                # 预处理阶段失败则直接返回 None（或可选择跳过）
                return None

            dist_vec, node_types_arr = cs.icdm_geodesic(tree, n_sample)  # <-- 可能抛超时异常
            return dist_vec, node_types_arr

        except Exception as e:
            last_exc = e
            try_id += 1
            if try_id > max_retries:
                break
            # 生成扰动副本并加入尝试列表
            if tmp_dir is None:
                tmp_dir = os.path.join(os.path.dirname(file_path), "_tmp_geodesic_safe")
            os.makedirs(tmp_dir, exist_ok=True)
            dst_path = os.path.join(
                tmp_dir,
                f"jitter_{os.path.basename(file_path)}.{try_id}.swc"
            )
            jitter_and_rescale_swc_file(
                src_path=file_path if try_id == 1 else tried_paths[try_id-1],
                dst_path=dst_path,
                scale_eps=scale_eps,
                coord_eps=coord_eps
            )
            tried_paths.append(dst_path)
            cleanup_paths.append(dst_path)

    # 清理副本
    for p in cleanup_paths:
        try:
            os.remove(p)
        except Exception:
            pass
    # 若创建了临时目录且为空，清掉
    if tmp_dir and os.path.isdir(tmp_dir):
        try:
            if not os.listdir(tmp_dir):
                os.rmdir(tmp_dir)
        except Exception:
            pass

    # 所有尝试失败
    return None

def run(
    infolder,
    out_csv,
    n_sample=100,
    types=(1,3,4),
    max_retries=10,
    scale_eps=1e-2,
    coord_eps=1e-6
):
    swc_files = sorted(glob.glob(os.path.join(infolder, "*.swc")))
    if not swc_files:
        raise RuntimeError(f"No .swc files found in {infolder}")

    # 写 CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        write_header(writer, n_sample)

        failed = []
        for i, fp in enumerate(swc_files, 1):
            base = os.path.basename(fp)
            sys.stdout.write(f"\r[{i}/{len(swc_files)}] {base} ...")
            sys.stdout.flush()

            res = geodesic_icdm_for_file(
                fp,
                n_sample=n_sample,
                types=types,
                max_retries=max_retries,
                scale_eps=scale_eps,
                coord_eps=coord_eps
            )
            if res is None:
                failed.append(base)
                continue

            dist_vec, _ = res
            # dist_vec 长度应为 C(n_sample, 2)
            if dist_vec.shape[0] != (n_sample * (n_sample - 1)) // 2:
                # 极罕见异常，记为失败
                failed.append(base)
                continue

            row = [base] + dist_vec.tolist()
            writer.writerow(row)

    print("\nDone.")
    if failed:
        print(f"Failed ({len(failed)}):")
        for name in failed[:20]:
            print("  -", name)
        if len(failed) > 20:
            print("  ...")

if __name__ == "__main__":
    # 示例调用：按你当前参数来
    # 修改 infolder / out_csv 路径即可
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infolder", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--n_sample", type=int, default=100)
    parser.add_argument("--max_retries", type=int, default=10)
    parser.add_argument("--scale_eps", type=float, default=1e-2)
    parser.add_argument("--coord_eps", type=float, default=1e-6)
    parser.add_argument("--types", type=int, nargs="+", default=[1,3,4])
    args = parser.parse_args()

    run(
        infolder=args.infolder,
        out_csv=args.out_csv,
        n_sample=args.n_sample,
        types=tuple(args.types),
        max_retries=args.max_retries,
        scale_eps=args.scale_eps,
        coord_eps=args.coord_eps
    )
