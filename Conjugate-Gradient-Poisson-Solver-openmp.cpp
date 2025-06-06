// File: Conjugate-Gradient-Poisson-Solver-openmp.cpp
// 使用 OpenMP 簡單平行化的 2D Poisson（CG 與 SOR）

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <omp.h>

#define ROW 51
#define COL 51
#define pi 3.141592
#define itmax 100000

// 函式宣告
void initialization(double **p);
void write_u(const char *dir_nm, const char *file_nm, double **p, double dx, double dy);
void SOR(double **p, double dx, double dy, double tol, double omega, double *tot_time, int *iter, int BC);
void Conjugate_Gradient(double **p, double dx, double dy, double tol, double *tot_time, int *iter, int BC);
double func(int i, int j, double dx, double dy);
void func_anal(double **p, int row_num, int col_num, double dx, double dy);
void error_rms(double **p, double **p_anal, double *err);
void poisson_solver(double **u, double **u_anal, double tol, double omega, int BC, int method, const char *dir_name);

int main(void) {
    double **u, **u_anal;
    char *dir_name;
    int i, BC;
    double tol, omega;

    // 建立結果資料夾
    system("mkdir -p RESULT");

    // 分配記憶體
    u      = (double **) malloc(ROW * sizeof(double *));
    u_anal = (double **) malloc(ROW * sizeof(double *));
    for (i = 0; i < ROW; i++) {
        u[i]      = (double *) malloc(COL * sizeof(double));
        u_anal[i] = (double *) malloc(COL * sizeof(double));
    }

    // 初始參數
    tol = 1e-6;
    omega = 1.8;
    dir_name = (char *) "./RESULT/";

    printf("\n---------------------------------------- \n");
    printf("Nx : %d, Ny : %d\n", ROW, COL);
    printf("Tolerance : %f, Omega : %f \n", tol, omega);
    printf("---------------------------------------- \n\n");

    BC = 1;

    // CG method
    printf("\n[CG Method]\n");
    poisson_solver(u, u_anal, tol, omega, BC, 1, dir_name);

    // SOR method
    printf("\n[SOR Method]\n");
    poisson_solver(u, u_anal, tol, omega, BC, 2, dir_name);

    // 釋放記憶體
    for (i = 0; i < ROW; i++) {
        free(u[i]);
        free(u_anal[i]);
    }
    free(u);
    free(u_anal);

    return 0;
}

// Poisson solver wrapper：計算解析解、呼叫 CG 或 SOR、輸出結果
void poisson_solver(double **u, double **u_anal, double tol, double omega, int BC, int method, const char *dir_name) {
    char file_name[128];
    double Lx = 1.0, Ly = 1.0;
    double dx = Lx / (ROW - 1), dy = Ly / (COL - 1);
    double err = 0.0, tot_time = 0.0;
    int iter = 0;

    // 先計算解析解並輸出
    func_anal(u_anal, ROW, COL, dx, dy);
    write_u(dir_name, "Analytic_solution.plt", u_anal, dx, dy);

    switch (method) {
        case 1:
            initialization(u);
            Conjugate_Gradient(u, dx, dy, tol, &tot_time, &iter, BC);
            error_rms(u, u_anal, &err);
            printf("CG method - Error : %e, Iteration : %d, Time : %f s\n", err, iter, tot_time);
            write_u(dir_name, "CG_result.plt", u, dx, dy);
            break;
        case 2:
            initialization(u);
            SOR(u, dx, dy, tol, omega, &tot_time, &iter, BC);
            error_rms(u, u_anal, &err);
            printf("SOR method - Error : %e, Iteration : %d, Time : %f s\n", err, iter, tot_time);
            write_u(dir_name, "SOR_result.plt", u, dx, dy);
            break;
    }
}

// RHS source function
double func(int i, int j, double dx, double dy) {
    return sin(pi * i * dx) * cos(pi * j * dy);
}

// 將矩陣 p 全部設為 0
void initialization(double **p) {
    int i, j;
    #pragma omp parallel for private(j)
    for (i = 0; i < ROW; i++) {
        for (j = 0; j < COL; j++) {
            p[i][j] = 0.0;
        }
    }
}

// 計算數值解與解析解的 RMS 誤差
void error_rms(double **p, double **p_anal, double *err) {
    int i, j;
    double sum = 0.0;
    #pragma omp parallel for private(j) reduction(+:sum)
    for (i = 0; i < ROW; i++) {
        for (j = 0; j < COL; j++) {
            double diff = p[i][j] - p_anal[i][j];
            sum += diff * diff;
        }
    }
    *err = sqrt(sum) / (ROW * COL);
}

// 計算解析解 u^a(i,j) = -1/(2*pi^2) * sin(pi x) cos(pi y)
void func_anal(double **p, int row_num, int col_num, double dx, double dy) {
    int i, j;
    #pragma omp parallel for private(j)
    for (i = 0; i < row_num; i++) {
        for (j = 0; j < col_num; j++) {
            p[i][j] = -1.0 / (2.0 * pi * pi) * sin(pi * i * dx) * cos(pi * j * dy);
        }
    }
}

// 將結果輸出成 .plt
void write_u(const char *dir_nm, const char *file_nm, double **p, double dx, double dy) {
    char file_path[128];
    sprintf(file_path, "%s%s", dir_nm, file_nm);
    FILE *stream = fopen(file_path, "w");
    fprintf(stream, "ZONE I=%d J=%d\n", ROW, COL);
    int i, j;
    for (i = 0; i < ROW; i++) {
        for (j = 0; j < COL; j++) {
            fprintf(stream, "%f %f %f\n", i * dx, j * dy, p[i][j]);
        }
        fprintf(stream, "\n");
    }
    fclose(stream);
}

// ----------------------------------------
//          SOR Method (Sequential Gauss-Seidel)
// 由於 SOR 內部每次更新與左方／上方格點有依賴
// 這裡只對「邊界條件」與「收斂量測」部分平行化，
// 主更新迴圈仍維持單緒執行以確保算法正確性。
// ----------------------------------------
void SOR(double **p, double dx, double dy, double tol, double omega, double *tot_time, int *iter, int BC) {
    double **p_new = (double **) malloc(ROW * sizeof(double *));
    for (int i = 0; i < ROW; i++) {
        p_new[i] = (double *) malloc(COL * sizeof(double));
    }
    initialization(p_new);

    double beta = dx / dy;
    clock_t start_t = clock();

    int i, j, it;
    for (it = 1; it < itmax; it++) {
        double SUM1 = 0.0, SUM2 = 0.0;

        // 主更新：依賴上一列 p_new[i-1][j] 和左方 p_new[i][j-1]
        for (i = 1; i < ROW - 1; i++) {
            for (j = 1; j < COL - 1; j++) {
                double temp = (p[i+1][j] + p_new[i-1][j]
                             + beta * beta * (p[i][j+1] + p_new[i][j-1])
                             - dx * dx * func(i, j, dx, dy))
                             / (2.0 * (1.0 + beta * beta));
                p_new[i][j] = p[i][j] + omega * (temp - p[i][j]);
            }
        }

        // 邊界條件：平行化（獨立操作）
        if (BC == 1) {
            #pragma omp parallel for
            for (j = 0; j < COL; j++) {
                p_new[0][j] = 0.0;
                p_new[ROW-1][j] = 0.0;
            }
            #pragma omp parallel for private(j)
            for (i = 0; i < ROW; i++) {
                p_new[i][0] = p_new[i][1];
                p_new[i][COL-1] = p_new[i][COL-2];
            }
        } else {
            #pragma omp parallel for
            for (j = 0; j < COL; j++) {
                p_new[0][j] = -1.0 / (2.0 * pi * pi) * func(0, j, dx, dy);
                p_new[ROW-1][j] = -1.0 / (2.0 * pi * pi) * func(ROW-1, j, dx, dy);
            }
            #pragma omp parallel for
            for (i = 0; i < ROW; i++) {
                p_new[i][0] = -1.0 / (2.0 * pi * pi) * func(i, 0, dx, dy);
                p_new[i][COL-1] = -1.0 / (2.0 * pi * pi) * func(i, COL-1, dx, dy);
            }
        }

        // 計算收斂量：平行化 reduce
        #pragma omp parallel for collapse(2) reduction(+:SUM1,SUM2) private(j)
        for (i = 1; i < ROW - 1; i++) {
            for (j = 1; j < COL - 1; j++) {
                SUM1 += fabs(p_new[i][j]);
                SUM2 += fabs(p_new[i+1][j] + p_new[i-1][j]
                           + beta * beta * (p_new[i][j+1] + p_new[i][j-1])
                           - (2.0 + 2.0 * beta * beta) * p_new[i][j]
                           - dx * dx * func(i, j, dx, dy));
            }
        }
        if (SUM2 / SUM1 < tol) {
            *iter = it;
            clock_t end_t = clock();
            *tot_time = double(end_t - start_t) / CLOCKS_PER_SEC;
            break;
        }

        // 更新 p
        for (i = 0; i < ROW; i++) {
            for (j = 0; j < COL; j++) {
                p[i][j] = p_new[i][j];
            }
        }
    }

    for (i = 0; i < ROW; i++) {
        free(p_new[i]);
    }
    free(p_new);
}

// ----------------------------------------
//      Conjugate Gradient (CG) 平行化版本
// ----------------------------------------
double norm_L2(double *a) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < ROW * COL; i++) {
        sum += a[i] * a[i];
    }
    return sqrt(sum);
}

double vvdot(double *a, double *b) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < ROW * COL; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void vmdot(double **A, double *x, double *b) {
    #pragma omp parallel for
    for (int i = 0; i < ROW * COL; i++) {
        double tmp = 0.0;
        for (int j = 0; j < ROW * COL; j++) {
            tmp += A[i][j] * x[j];
        }
        b[i] = tmp;
    }
}

void make_Abx(double **A, double *b, double *x, double **u, double dx, double dy) {
    int N = ROW * COL;
    // 初始化所有元素為 0
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = 0.0;
        }
    }
    // 組裝 A 與 b
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) {
            int idx = i * COL + j;
            if (i == 0 || i == ROW - 1 || j == 0 || j == COL - 1) {
                // 邊界
                A[idx][idx] = 1.0;
                b[idx] = 0.0;
            } else {
                // 內部點
                A[idx][idx] = -4.0;
                A[idx][idx - 1] = 1.0;
                A[idx][idx + 1] = 1.0;
                A[idx][idx - COL] = 1.0;
                A[idx][idx + COL] = 1.0;
                b[idx] = dx * dx * func(i, j, dx, dy);
            }
        }
    }
    // x 先設為 0
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        x[i] = 0.0;
    }
}

void Conjugate_Gradient(double **p, double dx, double dy, double tol, double *tot_time, int *iter, int BC) {
    int N = ROW * COL;
    double **A = (double **) malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        A[i] = (double *) malloc(N * sizeof(double));
    }
    double *tmp   = (double *) malloc(N * sizeof(double));
    double *x_vec = (double *) malloc(N * sizeof(double));
    double *b     = (double *) malloc(N * sizeof(double));
    double *z     = (double *) malloc(N * sizeof(double));
    double *r     = (double *) malloc(N * sizeof(double));
    double *r_new = (double *) malloc(N * sizeof(double));

    // 組裝 A, b, x_vec
    make_Abx(A, b, x_vec, p, dx, dy);

    // 初始化 r = b - A*x_vec (x_vec 初值為 0 → r = b)
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        r[i] = b[i];
        z[i] = r[i];
    }

    clock_t start_t = clock();
    for (int k = 0; k < itmax; k++) {
        // tmp = A * z
        vmdot(A, z, tmp);

        double zr = vvdot(r, r);
        double zAtmp = vvdot(z, tmp);
        double alpha = zr / zAtmp;

        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            x_vec[i] += alpha * z[i];
            r_new[i] = r[i] - alpha * tmp[i];
        }

        double norm_rnew = norm_L2(r_new);
        if (norm_rnew < tol) {
            // 把 x_vec 放回 p[][] 中
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < ROW; i++) {
                for (int j = 0; j < COL; j++) {
                    int idx = i * COL + j;
                    p[i][j] = x_vec[idx];
                }
            }
            *iter = k;
            clock_t end_t = clock();
            *tot_time = double(end_t - start_t) / CLOCKS_PER_SEC;
            break;
        }

        double beta = vvdot(r_new, r_new) / zr;
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            z[i] = r_new[i] + beta * z[i];
            r[i] = r_new[i];
        }
    }

    // 釋放記憶體
    for (int i = 0; i < N; i++) {
        free(A[i]);
    }
    free(A);
    free(tmp);
    free(x_vec);
    free(b);
    free(z);
    free(r);
    free(r_new);
}
