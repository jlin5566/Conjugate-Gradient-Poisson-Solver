# Ongoing works
- 正在把Conjugate-Gradient-Poisson-Solve.cpp和Conjugate-Gradient-Poisson-Solver_2nd.cpp合體
- 正在新增Accuracy and performance & performance scaling (i.e., wall-clock time vs. number of cells)功能進去plot_results.py
- plot_result.py 在terminal輸入 python3 plot_results.py --cut_y 0.1 沒反應，dubug中。發現圖沒有自動存檔。

# Checklist
- 你各位趕緊動起來做事阿。 5/24
- Please use the Fork→Pull→Push→Pull Request→Merge workflow. This is the requirement of the final project. 6/5

### Program
- [ ] **Conjugate gradient method to solve the 2D Poisson equation**
  - [x] CSR (Compressed Sparse Row) matrix **羅文鴻**
  - [ ] MKL
- [ ] **SOR method to solve the 2D Poisson equation**
  - [ ] Odd-even ordering
- [ ] **Hybrid Parallel Programming**
  - [ ] OpenMP
  - [ ] MPI
  - [ ] GPU
- [ ] **Performance**
  - [ ] Accuracy and performance & performance scaling (i.e., wall-clock time vs. number of cells)
  - [x] Compare the performance with SOR **蔡佾倫**

### Slides & presentation
- [ ] **Principles**
- [ ] **Implementation**
  - [ ] Conjugate gradient method
    - [ ] CSR (Compressed Sparse Row) matrix
    - [ ] MKL
  - [ ] Hybrid Parallel Programming
    - [ ] OpenMP
    - [ ] MPI
    - [ ] GPU
- [ ] **Performance**
  - [ ] Accuracy and performance & performance scaling (i.e., wall-clock time vs. number of cells)
  - [ ] Compare the performance with SOR
  - [ ] Performance with and without parallelization

### Division of work
  - OpenMP & MPI **林銘峻**
  - Accuracy and performance & performance scaling (i.e., wall-clock time vs. number of cells) **林傑澄**
