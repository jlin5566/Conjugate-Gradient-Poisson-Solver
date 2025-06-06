#!/usr/bin/env python3
# File: plot_mpi_cg_multi_grid.py

import os
import re
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import csv

# ———— 配置部分 ————

# 要测试的网格规模列表（ROW == COL），可按需增删
GRID_SIZES = [32, 64, 128, 256]

# 固定的 MPI 进程数
NUM_PROCS = 2

# 源文件名（MPI 版 CG 求解器）
SRC_FILE = "Conjugate-Gradient-Poisson-Solver-mpi.cpp"

# 可执行文件基名（脚本会针对每个 grid 重新编译为 solver_mpi_<grid>）
EXE_BASE = "solver_mpi"

# 用于保存编译产物和日志的临时目录
TMP_DIR = "tmp_mpi_benchmark"
os.makedirs(TMP_DIR, exist_ok=True)

# ———— 踩坑说明 ————
# 由于 MPI 输出中科学计数法可能会被断行，原先的 ([0-9Ee\.\-]+) 无法
# 完整匹配到像 7.845465e+02 这样的形式，导致只捕获到 7.845465e 而剩余的 +02 被丢失。
# 因此需要用更严格的正则，确保“mantissa + e/E + 可选 + or - + 指数数字”完整匹配。
LINE_PATTERN = re.compile(
    r"ITER\s+(\d+)\s+RESIDUAL\s+([0-9]+\.[0-9]+[eE][\+\-]?\d+)\s+TIME\s+([0-9]+\.[0-9]+[eE][\+\-]?\d+)"
)

# ———— 编译并生成可执行的函数 ————
def compile_with_grid(grid):
    """
    使用 `-DROW=grid -DCOL=grid` 宏重新编译 MPI 版代码，
    可执行文件命名为 f"{EXE_BASE}_{grid}"，放在 TMP_DIR 下。
    返回可执行文件的完整路径。
    """
    exe_name = f"{EXE_BASE}_{grid}"
    exe_path = os.path.join(TMP_DIR, exe_name)

    compile_cmd = [
        "mpic++", "-O2",
        f"-DROW={grid}", f"-DCOL={grid}",
        "-o", exe_path, SRC_FILE, "-lm"
    ]
    print(f"[Compile] grid={grid}×{grid} → {exe_name}")
    res = subprocess.run(compile_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print("====== 编译失败 ======")
        print(res.stderr)
        raise RuntimeError(f"Failed to compile for grid={grid}")
    return exe_path

# ———— 运行可执行并提取每步迭代数据的函数 ————
def run_and_parse(exe_path, num_procs, grid):
    """
    用 mpirun -np num_procs exe_path 运行，
    然后把 stdout/stderr 都捕获下来，存档在 tmp_mpi_benchmark/log_grid_<grid>.txt
    并从里边提取每次迭代的 (iter, residual, time)，放到一个列表返回。
    """
    cmd = ["mpirun", "-np", str(num_procs), exe_path]
    print(f"[Run] grid={grid}, MPI processes={num_procs}")

    # 将输出重定向到一个 log 文件里
    logfile = os.path.join(TMP_DIR, f"log_grid_{grid}.txt")
    with open(logfile, "w") as fw:
        # 注意：我们把 stdout/stderr 都写入同一个文件，方便后面 grep
        subprocess.run(cmd, stdout=fw, stderr=subprocess.STDOUT, text=True)

    # 读取刚才的 log，去匹配 ITER 行
    data = []
    with open(logfile, "r") as fr:
        all_text = fr.read()
        # 打印前 20 行，帮助排查输出
        head = "\n".join(all_text.splitlines()[:20])
        print(f"--- 前 20 行输出（grid={grid}） ---")
        print(head)
        print("...")

        # 正式去匹配 “ITER  <num>  RESIDUAL  <sci>  TIME  <sci>”
        for line in all_text.splitlines():
            m = LINE_PATTERN.search(line)
            if m:
                it  = int(m.group(1))
                res = float(m.group(2))
                t   = float(m.group(3))
                data.append((it, res, t))

    if not data:
        print(f"[Warning] 在运行 grid={grid} 时，未匹配到任何迭代信息。请检查 log：{logfile}")
    else:
        last_it, last_res, last_t = data[-1]
        print(f"[Info] grid={grid} 共提取到 {len(data)} 条迭代记录，最后一条："
              f"ITER={last_it}, RES={last_res:.3e}, TIME={last_t:.3e}")
    print("------------------------------------------------------------\n")
    return data

# ———— 主流程：针对每个网格规模，编译 + 运行 + 收集 ———
records = {}

for grid in GRID_SIZES:
    # 1) 编译
    exe_path = compile_with_grid(grid)

    # 2) 运行并解析
    recs = run_and_parse(exe_path, NUM_PROCS, grid)
    records[grid] = recs

# ———— 绘图部分 ————

# 1) Residual vs Iteration（对数刻度）
plt.figure(figsize=(7,5))
for grid in GRID_SIZES:
    recs = records.get(grid, [])
    if not recs:
        continue
    iters     = np.array([r[0] for r in recs])
    residuals = np.array([r[1] for r in recs])
    plt.semilogy(iters, residuals, marker='o', label=f"N={grid}")

plt.xlabel("Iteration")
plt.ylabel("Residual (log scale)")
plt.title(f"Residual vs Iteration (MPI procs={NUM_PROCS})")
plt.grid(which='both', ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("mpi_cg_residual_vs_iter_multi_grid.png", dpi=200)
plt.show()

# 2) Cumulative Time vs Iteration
plt.figure(figsize=(7,5))
for grid in GRID_SIZES:
    recs = records.get(grid, [])
    if not recs:
        continue
    iters = np.array([r[0] for r in recs])
    times = np.array([r[2] for r in recs])
    plt.plot(iters, times, marker='s', label=f"N={grid}")

plt.xlabel("Iteration")
plt.ylabel("Cumulative Time (s)")
plt.title(f"Cumulative Time vs Iteration (MPI procs={NUM_PROCS})")
plt.grid(ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("mpi_cg_time_vs_iter_multi_grid.png", dpi=200)
plt.show()

# 3) （可选）将原始数据保存到 CSV 以便后续分析
with open("mpi_cg_iter_data_multi_grid.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Grid", "Iteration", "Residual", "CumulativeTime"])
    for grid in GRID_SIZES:
        recs = records.get(grid, [])
        for it, res, t in recs:
            writer.writerow([grid, it, res, t])

print("\n=== 脚本执行完毕，已生成：")
print("  - mpi_cg_residual_vs_iter_multi_grid.png")
print("  - mpi_cg_time_vs_iter_multi_grid.png")
print("  - mpi_cg_iter_data_multi_grid.csv")
