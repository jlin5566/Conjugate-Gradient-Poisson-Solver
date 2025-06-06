// File: Conjugate-Gradient-Poisson-Solver-mpi.cpp
// 描述：在原单机版上，增加 MPI 通信，使 CG 求解器在多进程间共享向量内积和矩阵-向量乘法结果。
//       并且改为用“相对残差”作为收敛准则，迭代过程中由 rank=0 打印 ITER / RESIDUAL / TIME。
// 编译示例：
//   mpic++ -O2 -DROW=64 -DCOL=64 -o solver_mpi_64 Conjugate-Gradient-Poisson-Solver-mpi.cpp -lm
// 运行示例：
//   mpirun -np 2 ./solver_mpi_64

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#ifndef ROW
#define ROW 51
#endif

#ifndef COL
#define COL 51
#endif
#define pi 3.141592
#define itmax 100000

// 函数声明
void initialization(double **p);
void write_u(const char *dir_nm, const char *file_nm, double **p, double dx, double dy);
void SOR(double **p, double dx, double dy, double tol, double omega,
         double *tot_time, int *iter, int BC);
void Conjugate_Gradient(double **p, double dx, double dy, double tol,
                        double *tot_time, int *iter, int BC, MPI_Comm comm);
double func(int i, int j, double dx, double dy);
void func_anal(double **p, int row_num, int col_num, double dx, double dy);
void error_rms(double **p, double **p_anal, int row_num, int col_num, double *err);

// ——————————————————————————————
// MPI 并行版：向量内积（Allreduce）
// ——————————————————————————————
double vvdot_mpi(double *a, double *b, int n, MPI_Comm comm) {
    double local_dot = 0.0;
    for (int i = 0; i < n; i++) {
        local_dot += a[i] * b[i];
    }
    double global_dot = 0.0;
    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global_dot;
}

// ——————————————————————————————————————————————————
// MPI 并行版：矩阵-向量乘法 b = A * x
// A 存储为“稠密五点差分矩阵”形式，各进程先独立计算 b_local，然后 Allreduce 合并到 b_global
// ——————————————————————————————————————————————————
void vmdot_mpi(double **A, double *x, double *b_local, double *b_global, int n, MPI_Comm comm) {
    // 本地计算：b_local = A * x
    for (int i = 0; i < n; i++) {
        b_local[i] = 0.0;
    }
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) {
            int row_idx = i * COL + j;
            // 稠密形式，遍历整行
            for (int k = 0; k < ROW; k++) {
                for (int l = 0; l < COL; l++) {
                    int col_idx = k * COL + l;
                    b_local[row_idx] += A[row_idx][col_idx] * x[col_idx];
                }
            }
        }
    }
    // MPI_Allreduce 将各进程 b_local 累加到 b_global
    MPI_Allreduce(b_local, b_global, n, MPI_DOUBLE, MPI_SUM, comm);
}

// ——————————————————————————————————————————————————
// Poisson 求解主函数：先计算解析解，然后根据 method 调用 CG 或 SOR，最后写出结果
// ——————————————————————————————————————————————————
void poisson_solver(double **u, double **u_anal, double tol, double omega,
                    int BC, int method, const char *dir_name, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    char file_name[64];
    int iter = 0;
    double Lx = 1.0, Ly = 1.0;
    double dx = Lx / (ROW - 1);
    double dy = Ly / (COL - 1);
    double err = 0.0, tot_time = 0.0;

    // 1) 解析解
    strcpy(file_name, "Analytic_solution_mpi.plt");
    func_anal(u_anal, ROW, COL, dx, dy);
    if (rank == 0) {
        write_u(dir_name, file_name, u_anal, dx, dy);
    }

    // 2) 数值求解
    switch (method) {
        case 1:
            // Conjugate Gradient，并行版本
            initialization(u);
            Conjugate_Gradient(u, dx, dy, tol, &tot_time, &iter, BC, comm);
            if (rank == 0) {
                error_rms(u, u_anal, ROW, COL, &err);
                printf("MPI CG - Error: %e, Iter = %d, Time = %f s\n", err, iter, tot_time);
                strcpy(file_name, "CG_result_mpi.plt");
                write_u(dir_name, file_name, u, dx, dy);
            }
            break;
        case 2:
            // SOR（保持单机逻辑，不并行）
            initialization(u);
            SOR(u, dx, dy, tol, omega, &tot_time, &iter, BC);
            if (rank == 0) {
                error_rms(u, u_anal, ROW, COL, &err);
                printf("MPI SOR (单机模式) - Error: %e, Iter = %d, Time = %f s\n", err, iter, tot_time);
                strcpy(file_name, "SOR_result_mpi.plt");
                write_u(dir_name, file_name, u, dx, dy);
            }
            break;
        default:
            if (rank == 0) {
                fprintf(stderr, "未定义的方法编号 %d\n", method);
            }
    }
}

// ——————————————————————————————————————————————————
// main：MPI 初始化 + 内存分配 + 调用 poisson_solver + 释放 + MPI_Finalize
// ——————————————————————————————————————————————————
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 由 rank=0 创建输出目录
    if (rank == 0) {
        system("mkdir -p RESULT");
    }
    MPI_Barrier(comm);

    // 动态分配二维数组 u 与 u_anal
    double **u      = (double **) malloc(ROW * sizeof(double *));
    double **u_anal = (double **) malloc(ROW * sizeof(double *));
    for (int i = 0; i < ROW; i++) {
        u[i]      = (double *) malloc(COL * sizeof(double));
        u_anal[i] = (double *) malloc(COL * sizeof(double));
    }

    double tol = 1e-6;
    double omega = 1.8;
    int BC = 1;  // 边界条件类型：1=Dirichlet+Neumann

    if (rank == 0) {
        printf("MPI 版本 Poisson 求解: ROW=%d, COL=%d, procs=%d\n", ROW, COL, size);
        printf("--------------------------------------------\n");
    }

    // ----- CG 求解 -----
    if (rank == 0) {
        printf("\n>>[MPI CG Method]\n");
    }
    poisson_solver(u, u_anal, tol, omega, BC, 1, "./RESULT/", comm);

    // ----- SOR 求解（单机模式，仅 rank=0 输出） -----
    if (rank == 0) {
        printf("\n>>[MPI SOR Method (单机模式)]\n");
    }
    poisson_solver(u, u_anal, tol, omega, BC, 2, "./RESULT/", comm);

    // 释放内存
    for (int i = 0; i < ROW; i++) {
        free(u[i]);
        free(u_anal[i]);
    }
    free(u);
    free(u_anal);

    MPI_Finalize();
    return 0;
}

// ——————————————————————————————————————————————————
// 以下为原单机版函数，仅作必要改动使其能在 MPI 下编译
// ——————————————————————————————————————————————————

void initialization(double **p) {
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) {
            p[i][j] = 0.0;
        }
    }
}

void func_anal(double **p, int row_num, int col_num, double dx, double dy) {
    for (int i = 0; i < row_num; i++) {
        for (int j = 0; j < col_num; j++) {
            double x = i * dx;
            double y = j * dy;
            p[i][j] = -1.0 / (2.0 * pi * pi) * sin(pi * x) * cos(pi * y);
        }
    }
}

double func(int i, int j, double dx, double dy) {
    double x = i * dx;
    double y = j * dy;
    return sin(pi * x) * cos(pi * y);
}

void write_u(const char *dir_nm, const char *file_nm, double **p, double dx, double dy) {
    char file_path[128];
    sprintf(file_path, "%s%s", dir_nm, file_nm);
    FILE *stream = fopen(file_path, "w");
    if (!stream) {
        fprintf(stderr, "无法打开文件 %s 写入！\n", file_path);
        return;
    }
    fprintf(stream, "ZONE I=%d J=%d\n", ROW, COL);
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) {
            double x = i * dx;
            double y = j * dy;
            fprintf(stream, "%f %f %f\n", x, y, p[i][j]);
        }
    }
    fclose(stream);
}

void error_rms(double **p, double **p_anal, int row_num, int col_num, double *err) {
    double sum = 0.0;
    for (int i = 0; i < row_num; i++) {
        for (int j = 0; j < col_num; j++) {
            double d = p[i][j] - p_anal[i][j];
            sum += d * d;
        }
    }
    *err = sqrt(sum) / (row_num * col_num);
}

// 单机版 SOR，不并行
void SOR(double **p, double dx, double dy, double tol, double omega,
         double *tot_time, int *iter, int BC) {
    int i, j, it;
    double beta, SUM1, SUM2;
    double **p_new;
    clock_t start_t = 0, end_t = 0;

    start_t = clock();
    beta = dx / dy;

    p_new = (double **) malloc(ROW * sizeof(double *));
    for (i = 0; i < ROW; i++) {
        p_new[i] = (double *) malloc(COL * sizeof(double));
    }
    initialization(p_new);

    for (it = 1; it < itmax; it++) {
        SUM1 = 0.0;
        SUM2 = 0.0;
        // 迭代更新
        for (i = 1; i < ROW - 1; i++) {
            for (j = 1; j < COL - 1; j++) {
                p_new[i][j] = (p[i + 1][j] + p_new[i - 1][j]
                            + beta * beta * (p[i][j + 1] + p_new[i][j - 1])
                            - dx * dx * func(i, j, dx, dy))
                            / (2.0 * (1.0 + beta * beta));
                p_new[i][j] = p[i][j] + omega * (p_new[i][j] - p[i][j]);
            }
        }
        // 边界条件
        if (BC == 1) {
            // 上下 Dirichlet = 0
            for (j = 0; j < COL; j++) {
                p_new[0][j]      = 0.0;
                p_new[ROW - 1][j] = 0.0;
            }
            // 左右 Neumann
            for (i = 0; i < ROW; i++) {
                p_new[i][0]       = p_new[i][1];
                p_new[i][COL - 1] = p_new[i][COL - 2];
            }
        } else {
            // BC == 2: 直接用解析函数赋值边界
            for (j = 0; j < COL; j++) {
                p_new[0][j]      = -1.0 / (2.0 * pi * pi) * func(0, j, dx, dy);
                p_new[ROW - 1][j] = -1.0 / (2.0 * pi * pi) * func(ROW - 1, j, dx, dy);
            }
            for (i = 0; i < ROW; i++) {
                p_new[i][0]       = -1.0 / (2.0 * pi * pi) * func(i, 0, dx, dy);
                p_new[i][COL - 1] = -1.0 / (2.0 * pi * pi) * func(i, COL - 1, dx, dy);
            }
        }
        // 收敛判断
        for (i = 1; i < ROW - 1; i++) {
            for (j = 1; j < COL - 1; j++) {
                SUM1 += fabs(p_new[i][j]);
                SUM2 += fabs(p_new[i + 1][j] + p_new[i - 1][j]
                             + beta * beta * (p_new[i][j + 1] + p_new[i][j - 1])
                             - (2.0 + 2.0 * beta * beta) * p_new[i][j]
                             - dx * dx * func(i, j, dx, dy));
            }
        }
        if (SUM2 / SUM1 < tol) {
            *iter = it;
            end_t = clock();
            *tot_time = (double)(end_t - start_t) / CLOCKS_PER_SEC;
            break;
        }
        // 更新 p
        for (i = 0; i < ROW; i++) {
            for (j = 0; j < COL; j++) {
                p[i][j] = p_new[i][j];
            }
        }
    }
    // 释放临时矩阵
    for (i = 0; i < ROW; i++) {
        free(p_new[i]);
    }
    free(p_new);
}

// ——————————————————————————————————————————————————
// 并行版 Conjugate Gradient，使用 MPI_Allreduce 来做向量内积与矩阵-向量乘法合并
// 收敛准则改为相对残差：||r|| / ||b|| < tol
// 并在每次迭代打印 ITER / RESIDUAL / TIME
// ——————————————————————————————————————————————————
void Conjugate_Gradient(double **p, double dx, double dy, double tol,
                        double *tot_time, int *iter, int BC, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int n = ROW * COL;

    // 1) 分配 A(稠密五点差分矩阵)、x、b、z、r、r_new、Ax_loc、Ax_glob
    double **A      = (double **) malloc(n * sizeof(double *));
    double *x       = (double *) malloc(n * sizeof(double));
    double *b       = (double *) malloc(n * sizeof(double));
    double *z       = (double *) malloc(n * sizeof(double));
    double *r       = (double *) malloc(n * sizeof(double));
    double *r_new   = (double *) malloc(n * sizeof(double));
    double *Ax_loc  = (double *) malloc(n * sizeof(double));
    double *Ax_glob = (double *) malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        A[i] = (double *) malloc(n * sizeof(double));
    }

    // ——————————————————————————————
    // 2) 构造矩阵 A（五点差分），对角 = +4，邻接 = −1，并考虑边界
    //    Poisson 离散后: 4u_{ij} − u_{i±1,j} − u_{i,j±1} = dx^2 f_{ij}
    // ——————————————————————————————
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) {
            int row_idx = i * COL + j;
            // 先把整行置 0
            for (int col_idx = 0; col_idx < n; col_idx++) {
                A[row_idx][col_idx] = 0.0;
            }
            // 上下边界：Dirichlet u=0，A[row,row] = 1
            if (i == 0 || i == ROW - 1) {
                A[row_idx][row_idx] = 1.0;
            }
            // 左右边界：Neumann du/dn = 0 对应 u[i][0] = u[i][1] → A[row,row] = 1, A[row,neighbor] = -1
            else if (j == 0 || j == COL - 1) {
                A[row_idx][row_idx] = 1.0;
                int neighbor = i * COL + (j == 0 ? 1 : (COL - 2));
                A[row_idx][neighbor] = -1.0;
            }
            // 内点：五点差分
            else {
                A[row_idx][row_idx] = 4.0;
                A[row_idx][(i - 1) * COL + j] = -1.0;   // 上
                A[row_idx][(i + 1) * COL + j] = -1.0;   // 下
                A[row_idx][i * COL + (j - 1)] = -1.0;   // 左
                A[row_idx][i * COL + (j + 1)] = -1.0;   // 右
            }
        }
    }

    // ——————————————————————————————
    // 3) 构造向量 b：b[idx] = dx^2 * f(i,j)，边界处 b=0
    // ——————————————————————————————
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) {
            int idx = i * COL + j;
            if (i == 0 || i == ROW - 1 || j == 0 || j == COL - 1) {
                b[idx] = 0.0;
            } else {
                b[idx] = dx * dx * func(i, j, dx, dy);
            }
        }
    }

    // ——————————————————————————————
    // 4) 计算 ||b||_2，并取平方根
    // ——————————————————————————————
    double local_b2 = 0.0;
    for (int idx = 0; idx < n; idx++) {
        local_b2 += b[idx] * b[idx];
    }
    double bnorm2 = 0.0;
    MPI_Allreduce(&local_b2, &bnorm2, 1, MPI_DOUBLE, MPI_SUM, comm);
    double bnorm = sqrt(bnorm2);

    // ——————————————————————————————
    // 5) 初始化：x = 0，r = b - A x = b，z = r
    // ——————————————————————————————
    for (int idx = 0; idx < n; idx++) {
        x[idx] = 0.0;
        r[idx] = b[idx];
        z[idx] = r[idx];
    }

    // ——————————————————————————————
    // 6) 计算初始 rr = r·r
    // ——————————————————————————————
    double rr = vvdot_mpi(r, r, n, comm);

    // ——————————————————————————————
    // 7) 相对残差的平方阈值： rr_new < tol^2 * bnorm2
    // ——————————————————————————————
    double tol2_rel = tol * tol * bnorm2;

    // ——————————————————————————————
    // 8) 主迭代：CG
    //    - 每次迭代后，rank=0 打印 “ITER   %d   RESIDUAL   %e   TIME   %e”
    // ——————————————————————————————
    double t_start = MPI_Wtime();
    int it;
    for (it = 0; it < itmax; it++) {
        // 8.1) 本地计算 Ax_loc = A * z，然后 Allreduce 得 Ax_glob
        vmdot_mpi(A, z, Ax_loc, Ax_glob, n, comm);

        // 8.2) alpha = (r·r)/(z·(A z))
        double zAz = vvdot_mpi(z, Ax_glob, n, comm);
        double alpha = rr / zAz;

        // 8.3) 更新 x, r_new = r - alpha * (A z)
        for (int idx = 0; idx < n; idx++) {
            x[idx]     += alpha * z[idx];
            r_new[idx]  = r[idx] - alpha * Ax_glob[idx];
        }

        // 8.4) rr_new = r_new·r_new
        double rr_new = vvdot_mpi(r_new, r_new, n, comm);

        // 8.5) rank=0 打印迭代信息：迭代次数、相对残差、累计时间
        if (rank == 0) {
            double cumtime = MPI_Wtime() - t_start;
            double rel_resid = sqrt(rr_new) / bnorm;
            printf("ITER   %6d   RESIDUAL   %.6e   TIME   %.6e\n",
                   it + 1, rel_resid, cumtime);
        }

        // 8.6) 收敛判断：||r_new||^2 < tol^2 * ||b||^2
        if (rr_new < tol2_rel) {
            // 将 x 写回二维数组 p[i][j]
            for (int ii = 0; ii < ROW; ii++) {
                for (int jj = 0; jj < COL; jj++) {
                    int idx2 = ii * COL + jj;
                    p[ii][jj] = x[idx2];
                }
            }
            rr = rr_new;
            break;
        }

        // 8.7) beta = rr_new / rr；更新 z, r
        double beta = rr_new / rr;
        for (int idx = 0; idx < n; idx++) {
            z[idx] = r_new[idx] + beta * z[idx];
            r[idx] = r_new[idx];
        }
        rr = rr_new;
    }
    double t_end = MPI_Wtime();
    *tot_time = t_end - t_start;
    *iter     = it + 1;

    // 9) 如果迭代结束时还没写回 x→p，再补写一次
    if (rr >= tol2_rel) {
        for (int ii = 0; ii < ROW; ii++) {
            for (int jj = 0; jj < COL; jj++) {
                int idx2 = ii * COL + jj;
                p[ii][jj] = x[idx2];
            }
        }
    }

    // 10) 释放内存
    for (int i = 0; i < n; i++) {
        free(A[i]);
    }
    free(A);
    free(x);
    free(b);
    free(z);
    free(r);
    free(r_new);
    free(Ax_loc);
    free(Ax_glob);
}
