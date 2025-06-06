#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark.py

自动化测试不同线程数与不同网格规模下
CG（Conjugate Gradient）解 Poisson 问题的并行效率

输出：
1) Speed-up vs Number of Threads
2) Efficiency vs Number of Threads
3) Parallel Efficiency vs Grid Size

注意：需要预先安装 numpy 与 matplotlib。如果无法通过 pip 安装，
请考虑创建虚拟环境 (venv) 或者使用系统包管理器 apt 安装
python3-numpy、python3-matplotlib 等。
"""

import os
import re
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import csv

# ———— 配置部分 ————

# 需要测试的线程数列表 (如果不包含 1，会自动补充)
THREAD_LIST = [2, 4, 8, 16]

# 需要测试的网格规模 (ROW == COL)，可以自己增删
GRID_SIZES = [51, 101, 201, 401]

# 源文件名
SRC_FILE = "Conjugate-Gradient-Poisson-Solver-openmp.cpp"

# 可执行文件基名 (脚本会针对每种 grid 重新编译为 solver_openmp_<grid>)
EXE_BASE = "solver_openmp"

# 临时目录，用于存放各编译产物
TMP_DIR = "tmp_benchmark"
os.makedirs(TMP_DIR, exist_ok=True)

# 正则表达式，用于从 solver 输出中提取 CG 方法的 “Time : xxx s”
# 假设原代码输出格式类似：
#    CG method - Error : 1.2345e-07, Iteration : 123, Time : 0.012345 s
CG_TIME_PATTERN = re.compile(r"CG method.*Time\s*:\s*([0-9]+\.[0-9]+)\s*s")

# ———— 编译＋运行＋提取时序的函数 ————

def compile_with_grid(grid):
    """
    使用 `-DROW=grid -DCOL=grid` 宏重新编译 OpenMP 代码，
    可执行文件命名为 f"{EXE_BASE}_{grid}"。
    返回编译好的可执行文件路径。
    """
    exe_name = f"{EXE_BASE}_{grid}"
    exe_path = os.path.join(TMP_DIR, exe_name)
    compile_cmd = [
        "g++", "-O2", "-fopenmp",
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

def run_and_get_cg_time(exe_path, num_threads):
    """
    在环境变量 OMP_NUM_THREADS=num_threads 下执行 exe_path，
    并从 stdout 中提取 CG 时间（单位：秒）。返回浮点数。
    """
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(num_threads)
    proc = subprocess.run([exe_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    out = proc.stdout + "\n" + proc.stderr
    m = CG_TIME_PATTERN.search(out)
    if not m:
        print("=== WARNING: 未能从输出中提取 CG 时间 ===")
        print("程序输出：")
        print(out)
        raise RuntimeError("Cannot parse CG time")
    return float(m.group(1))

# ———— 主流程：逐网格规模、逐线程数 测试，收集数据 ————

# 确保 1 一定出现在线程列表里
if 1 not in THREAD_LIST:
    THREAD_LIST.insert(0, 1)

# times[grid][threads] = CG 时间 (秒)
times = {grid: {} for grid in GRID_SIZES}

for grid in GRID_SIZES:
    # 1) 编译
    exe_path = compile_with_grid(grid)

    # 2) 针对每个线程数运行
    for th in THREAD_LIST:
        print(f"[Run] grid={grid}, threads={th}")
        t_cg = run_and_get_cg_time(exe_path, th)
        times[grid][th] = t_cg
        print(f"    CG time = {t_cg:.6f} s")

# ———— 计算加速比与效率 ————

# speedup[grid][threads] = T(1)/T(threads)
speedup   = {grid: {} for grid in GRID_SIZES}
efficiency= {grid: {} for grid in GRID_SIZES}

for grid in GRID_SIZES:
    T1 = times[grid][1]
    for th in THREAD_LIST:
        speedup[grid][th]    = T1 / times[grid][th]
        efficiency[grid][th] = speedup[grid][th] / th

# ———— 绘图部分 ————

# 1) Speed-up vs Threads（固定最大网格）
plt.figure(figsize=(6,4))
grid0 = GRID_SIZES[-1]
x = np.array(THREAD_LIST)
y_speed = np.array([speedup[grid0][th] for th in THREAD_LIST])
plt.plot(x, y_speed, marker='o', label=f"Grid {grid0}×{grid0}")
plt.xlabel("Number of Threads")
plt.ylabel("Speed-up (T(1)/T(n))")
plt.title("Speed-up vs Number of Threads")
plt.xticks(x)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("speedup_vs_threads.png", dpi=200)
plt.show()

# 2) Efficiency vs Threads（同样固定网格）
plt.figure(figsize=(6,4))
y_eff = np.array([efficiency[grid0][th] for th in THREAD_LIST])
plt.plot(x, y_eff, marker='s', label=f"Grid {grid0}×{grid0}")
plt.xlabel("Number of Threads")
plt.ylabel("Efficiency = Speed-up / Threads")
plt.title("Efficiency vs Number of Threads")
plt.xticks(x)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("efficiency_vs_threads.png", dpi=200)
plt.show()

# 3) Parallel Efficiency vs Grid Size（选固定线程数，比如 THREAD_LIST[-1]）
fixed_th = THREAD_LIST[-1]
x_grids = np.array(GRID_SIZES)
y_par_eff = []
for grid in GRID_SIZES:
    T1 = times[grid][1]
    Tn = times[grid][fixed_th]
    pe = (T1 / Tn) / fixed_th
    y_par_eff.append(pe)
y_par_eff = np.array(y_par_eff)

plt.figure(figsize=(6,4))
plt.plot(x_grids, y_par_eff, marker='^')
plt.xlabel("Grid Size (N × N)")
plt.ylabel(f"Parallel Efficiency @ {fixed_th} Threads")
plt.title("Parallel Efficiency vs Grid Size")
plt.xticks(x_grids)
plt.grid(True)
plt.tight_layout()
plt.savefig("paralleleff_vs_gridsize.png", dpi=200)
plt.show()

# ———— 把数据存成 CSV，以便以后分析 ————
with open("benchmark_results.csv", "w", newline='') as f:
    writer = csv.writer(f)
    # 写 header
    header = ["Grid"]
    for th in THREAD_LIST:
        header.append(f"T({th}) (s)")
    for th in THREAD_LIST:
        header.append(f"Speedup@{th}")
    for th in THREAD_LIST:
        header.append(f"Eff@{th}")
    writer.writerow(header)

    # 写每行
    for grid in GRID_SIZES:
        row = [grid]
        # T(n)
        for th in THREAD_LIST:
            row.append(f"{times[grid][th]:.6f}")
        # Speed-up(n)
        for th in THREAD_LIST:
            row.append(f"{speedup[grid][th]:.6f}")
        # Efficiency(n)
        for th in THREAD_LIST:
            row.append(f"{efficiency[grid][th]:.6f}")
        writer.writerow(row)

print("\n=== Benchmark 完成，已生成 ===")
print("  - speedup_vs_threads.png")
print("  - efficiency_vs_threads.png")
print("  - paralleleff_vs_gridsize.png")
print("  - benchmark_results.csv")
