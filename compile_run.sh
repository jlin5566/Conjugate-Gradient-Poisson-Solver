#!/bin/bash

# ª½±µ¥Î g++ ½sÄ¶
#gcc -O2 -o poisson_solver Conjugate-Gradient-Poisson-Solver.c -lm
g++ -O2 -o poisson_solver Conjugate-Gradient-Poisson-Solver.c -lm

# ¨Ï¥Î¤è¦¡./poisson_solver BC use_matrix_A Nx Ny

./poisson_solver 2 false 100 100
