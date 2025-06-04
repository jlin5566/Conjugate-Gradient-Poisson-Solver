import numpy as np
import matplotlib.pyplot as plt
import argparse

# ---------------------------------------------------------
# 解析命令列參數
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Plot CG, SOR, and Analytic solution comparison at given y.")
parser.add_argument("--cut_y", type=float, required=True, help="Y value to cut the slice")
args = parser.parse_args()
cut_y = args.cut_y

# ---------------------------------------------------------
# 讀取 .plt 檔案資料
# ---------------------------------------------------------
def read_plt(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()[1:]  # 跳過第一行 ZONE I=... J=...
    data = np.array([[float(val) for val in line.split()] for line in lines])
    return data[:, 0], data[:, 1], data[:, 2]

# ---------------------------------------------------------
# 主程式：畫圖與誤差分析
# ---------------------------------------------------------
# 讀取三個解
x_a, y_a, z_a = read_plt("RESULT/Analytic_solution.plt")
x_cg, y_cg, z_cg = read_plt("RESULT/CG_result.plt")
x_sor, y_sor, z_sor = read_plt("RESULT/SOR_result.plt")

# 選擇 y = cut_y 的橫切線
mask_a = np.isclose(y_a, cut_y, atol=1e-2)
mask_cg = np.isclose(y_cg, cut_y, atol=1e-2)
mask_sor = np.isclose(y_sor, cut_y, atol=1e-2)

# ---------------------------------------------------------
# 圖一：解的比較圖
# ---------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(x_a[mask_a], z_a[mask_a], label="Analytic", linewidth=2)
plt.plot(x_cg[mask_cg], z_cg[mask_cg], label="CG", linestyle='--')
plt.plot(x_sor[mask_sor], z_sor[mask_sor], label="SOR", linestyle=':')

# 設定 y 軸範圍
ymin = min(np.min(z_a[mask_a]), np.min(z_cg[mask_cg]), np.min(z_sor[mask_sor]))
ymax = max(np.max(z_a[mask_a]), np.max(z_cg[mask_cg]), np.max(z_sor[mask_sor]))
margin = 0.1 * (ymax - ymin)
plt.ylim(ymin - margin, ymax + margin)

plt.xlabel("x")
plt.ylabel(f"u(x, y={cut_y})")
plt.title(f"Comparison of Analytic, CG, and SOR Solutions at y = {cut_y}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 圖二：誤差圖（CG 與 SOR 對解析解）
# ---------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(x_a[mask_a], z_cg[mask_cg] - z_a[mask_a], label="CG Error", linestyle='--')
plt.plot(x_a[mask_a], z_sor[mask_sor] - z_a[mask_a], label="SOR Error", linestyle=':')

plt.xlabel("x")
plt.ylabel("Error")
plt.title(f"Error of CG and SOR Compared to Analytic at y = {cut_y}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
