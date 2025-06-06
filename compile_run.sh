#!/bin/bash

# 直接用 g++ 編譯
#gcc -O2 -o poisson_solver Conjugate-Gradient-Poisson-Solver.c -lm
g++ -O2 -o poisson_solver Conjugate-Gradient-Poisson-Solver.c -lm

# 使用方式./poisson_solver BC use_matrix_A Nx Ny
./poisson_solver 2 false 50 50
./poisson_solver 2 false 100 100
./poisson_solver 2 false 200 200
