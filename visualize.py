import numpy as np
import matplotlib.pyplot as plt

filenames = [
    ("RESULT/Analytic_solution.plt", "Analytic Solution"),
    ("RESULT/CG_result_1.plt",    "CG_1 Solution"),
    ("RESULT/CG_result_2.plt",    "CG_2 Solution"),
    ("RESULT/SOR_result_1.plt",   "SOR_1 Solution"),
    ("RESULT/SOR_result_2.plt",   "SOR_2 Solution")
]

def read_plt(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    header = lines[0].split()
    ROW = int(header[1].split('=')[1])
    COL = int(header[2].split('=')[1])
    data = np.loadtxt(lines[1:])
    x = data[:, 0].reshape((ROW, COL))
    y = data[:, 1].reshape((ROW, COL))
    u = data[:, 2].reshape((ROW, COL))
    return x, y, u, ROW


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (fname, title) in enumerate(filenames):
    ax = axes[idx]
    x, y, u, ROW= read_plt(fname)

    cf = ax.contourf(x, y, u, levels=50, cmap='jet')
    ax.set_title(f"{title} for N = {ROW}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    
    cbar = fig.colorbar(cf, ax=ax, shrink=0.8)
    cbar.set_label("u(x,y)")

axes[-1].axis("off")

plt.tight_layout()
plt.savefig(f"Visualization.png", dpi=300)
plt.show()
