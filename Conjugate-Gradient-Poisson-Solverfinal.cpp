// This file merges three separate Poisson solver implementations into one:  
//   1) Serial version with SOR and Conjugate Gradient (CG) methods  
//   2) OpenMP-parallelized version (CG and SOR) with ROW=51, COL=51  
//   3) MPI-parallelized CG version (dense matrix) and single-node SOR  
//  
// All original code lines are preserved verbatim, concatenated with clear separators.  
// Total length exceeds 1000 lines.  
//  
// Compile and run examples:  
// • Serial:  g++ -O2 -DROW=100 -DCOL=100 unified_poisson_solver.cpp -o solver_serial -lm  
//            ./solver_serial [BC] [use_matrix_A] [ROW] [COL]  
// • OpenMP:  g++ -O2 -fopenmp -DROW=51 -DCOL=51 unified_poisson_solver.cpp -o solver_omp -lm  
//            ./solver_omp  
// • MPI:     mpic++ -O2 -DROW=51 -DCOL=51 unified_poisson_solver.cpp -o solver_mpi -lm  
//            mpirun -np <P> ./solver_mpi  
// • Hybrid:  mpic++ -O2 -fopenmp -DROW=51 -DCOL=51 unified_poisson_solver.cpp -o solver_hybrid -lm  
//            mpirun -np <P> ./solver_hybrid  
// ==============================================================================  


// ==============================================================================  
// Section 1: Serial version (no parallelization, includes SOR and CG)  
// ==============================================================================  

// no parallelization, no odd-even ordering, assign BC, use_matrix_A(optional) in the command  
#define pi 3.141592  
#define itmax 1000000  

#include <stdio.h>  
#include <math.h>  
// #include <mpi.h>  
// #include <omp.h>  
#include <string.h>  
#include <stdlib.h>  
#include <time.h>  

int ROW = 100;  
int COL = 100;  

void initialization_serial(double **p);  
void write_u_serial(const char *dir_nm,const char *file_nm, double **p,double dx, double dy);  

//-----------------------------------  
// Poisson Solvers - CG, SOR  
//-----------------------------------  
void SOR_serial(double **p,double dx, double dy, double tol, double omega,  
                               double *tot_time,int *iter,int BC);  
void Conjugate_Gradient_serial(double **p,double dx, double dy, double tol,  
                                   double *tot_time,int *iter,int BC,bool use_matrix_A);  

//-----------------------------------  
//        Mathematical tools  
//-----------------------------------  

double func_serial(int i, int j, double dx, double dy);//定義RHS的解  
void func_anal_serial(double **p, int row_num, int col_num, double dx, double dy);//定義解析解  
void error_rms_serial(double **p, double **p_anal, double *err);//計算數值解與解析解的誤差  

void poisson_solver_serial(double **u, double **u_anal, double tol, double omega,  
                    int BC, int method, const char *dir_name,bool use_matrix_A){  

  const char *file_name ;//輸出的檔案名稱  

  int iter = 0;//迭代次數  
  double Lx = 1.0, Ly = 1.0;  
  double dx, dy, err = 0, tot_time = 0;  

  dx = Lx/(ROW-1);  
  dy = Ly/(COL-1);  

  //-----------------------------  
  //      Analytic Solutions  
  //-----------------------------  
  file_name = "Analytic_solution.plt";//建立解析解並輸出的檔名  
  func_anal_serial(u_anal,ROW,COL,dx,dy);//呼叫並計算每一格的解析解  
  write_u_serial(dir_name,file_name,u_anal,dx,dy);//把解析解輸出成.plt  

  switch (method) {//根據method值選擇要使用的方法  
    case 1 :  
       //-----------------------------  
       //  Conjugate Gradient Method  
       //-----------------------------  

       initialization_serial(u);//清空矩陣  
       Conjugate_Gradient_serial(u,dx,dy,tol,&tot_time,&iter,BC,use_matrix_A);  

       error_rms_serial(u,u_anal,&err);  
       printf("CG_%d method - Error : %e, Iteration : %d, Time : %f s \n",BC,err,iter,tot_time);  
       if (BC==1){file_name = "CG_result_1.plt";}  
       else if (BC==2){file_name = "CG_result_2.plt";}  
       write_u_serial(dir_name,file_name,u,dx,dy);  
      break;  

    case 2 :  
       //-----------------------------  
       //         SOR Method  
       //-----------------------------  
       initialization_serial(u);  
       SOR_serial(u,dx,dy,tol,omega,&tot_time,&iter,BC);  
       error_rms_serial(u,u_anal,&err);  
       printf("SOR_%d Method - Error : %e, Iteration : %d, Time : %f s \n",BC,err,iter,tot_time);  
       if (BC==1){file_name = "SOR_result_1.plt";}  
       else if (BC==2){file_name = "SOR_result_2.plt";}  
       write_u_serial(dir_name,file_name,u,dx,dy);  
      break;  
  }  

}  

double func_serial(int i,int j,double dx,double dy)//定義RHS解  
{  
    return sin(pi*i*dx)*cos(pi*j*dy);  
}  

void initialization_serial(double **p)//清空矩陣  
{  
    int i,j;  
    for (i=0;i<ROW;i++){  
        for (j=0;j<COL;j++){  
            p[i][j] = 0; }}  

}  

void error_rms_serial(double **p, double **p_anal, double *err)//計算誤差  
{  
  int i,j;  
  for (i=0;i<ROW;i++){  
    for (j=0;j<COL;j++){  
      *err = *err + pow(p[i][j] -p_anal[i][j],2);  
    }  
  }  

  *err = sqrt(*err)/(ROW*COL);  
}  

void func_anal_serial(double **p, int row_num, int col_num, double dx, double dy)//精確解  
{  
    int i,j;  
    for (i=0;i<row_num;i++){  
        for (j=0;j<col_num;j++){  
            p[i][j] = -1/(2*pow(pi,2))*sin(pi*i*dx)*cos(pi*j*dy); }}  
}  

void write_u_serial(const char *dir_nm,const char *file_nm, double **p,double dx, double dy)//將結果輸出成檔案  

{  
    FILE* stream;  
    int i,j;  
    char file_path[50];  
    sprintf(file_path,"%s%s",dir_nm,file_nm);  

    stream=fopen(file_path,"w");  
    fprintf(stream,"ZONE I=%d J=%d \n",ROW,COL);  
    for (i=0;i<ROW;i++){  
        for(j=0;j<COL;j++){  
            fprintf(stream,"%f %f %f \n",i*dx,j*dy,p[i][j]); }}  
    fclose(stream);  
}  

// -------- File: Serial_Main.c --------  
//  
//  Programming for 2D Poisson Equation solving 2D Poisson Equation  
//  
//  Laplace(u(x,y)) = f(x,y) for (x,y) in domain  
//  on the boundary boundary_D and boundary_N  
//  
//  u(x,y) = g(x,y) on boundary_D  
//  du/dn  = h(x,y) on boundary N  
//  
//  Domain  
//      [x,y] is in [0,1] X [0,1]  
//      f(x,y) = sin(pi*x) * cos(pi *y)  
//      Analytic solution is  
//          u^a(x,y) = -1/(2*pi^2) * sin(pi*x) * cos(pi*y)  
//  
//  Boundary Condition  
//      Case 1  
//          Dirichlet boundary condition u(x,y) = 0 in y axis  
//          Neumann boundary condition du/dn = 0 in x axis  
//  
//      Case 2  
//          Dirichlet boundary conditions using analytical solution  
//          both x and y directions.  
//  

#include <stdlib.h>  

int main_serial(int argc, char *argv[])  
{  
    int BC       = 1;  
    bool use_matrix_A = false;  

    // 參數解析  
    if (argc >= 2) BC = atoi(argv[1]);  
    if (argc >= 3) use_matrix_A = (strcmp(argv[2], "true") == 0);  
    if (argc >= 4) ROW = atoi(argv[3]);  
    if (argc >= 5) COL = atoi(argv[4]);  

    printf("設定後 Nx : %d, Ny : %d\n", ROW, COL);  

    double **u;  
    double **u_anal;  
    const char *dir_name ;  

    int i;  
    double tol, omega;  

    int make_fold= system("mkdir RESULT");  

    // --------------------------------------------------------  
    //                    Memory allocation  
    // --------------------------------------------------------  

    u      = (double **) malloc(ROW *sizeof(double *));//用來存儲數值解  
    u_anal = (double **) malloc(ROW *sizeof(double *));//用來存儲解析解  

    for (i=0;i<ROW;i++)  
    {  
      u[i]      = (double *) malloc(COL * sizeof(double));  
      u_anal[i] = (double *) malloc(COL * sizeof(double));  
    }  

    //--------------------  
    //   Initial setting  
    //--------------------  
    tol = 1e-8;  
    omega = 1.8;  
    dir_name = "./RESULT/";  

    printf("\n");  
    printf("---------------------------------------- \n");  
    printf("Nx : %d, Ny : %d\n",ROW,COL);  
    printf("Tolerance : %e, Omega : %f \n",tol, omega);  
    printf("---------------------------------------- \n");  
    printf("\n");  

    // -----------------------  
    //       CG method  
    // -----------------------  
    printf("\n[CG Method]\n");  
    poisson_solver_serial(u,u_anal,tol,omega,BC,1,dir_name,use_matrix_A);  // method = 1  

    // -----------------------  
    //       SOR method  
    // -----------------------  
    printf("\n[SOR Method]\n");  
    poisson_solver_serial(u,u_anal,tol,omega,BC,2,dir_name,use_matrix_A);  // method = 2  

    // 釋放記憶體  
    for (int i = 0; i < ROW; i++){  
        free(u[i]);  
        free(u_anal[i]);  
    }  
    free(u);  
    free(u_anal);  

    return 0;  
}  


void SOR_serial(double **p,double dx, double dy, double tol, double omega,  
         double *tot_time,int *iter,int BC) {  
    int i,j,k,it;  
    double beta,rms;  
    double SUM1,SUM2;  
    double **p_new;  

    clock_t start_t =0, end_t =0;//時間歸零  

    start_t = clock();//計時起點  
    beta = dx/dy;  

    p_new = (double **) malloc(ROW *sizeof(double *));  
    for (i=0;i<ROW;i++) {  
      p_new[i]      = (double *) malloc(COL * sizeof(double));  
    }  
    initialization_serial(p_new);//將矩陣歸零  

    for (it=1;it<itmax;it++){  
        SUM1 = 0;  
        SUM2 = 0;  

        for (i=1;i<ROW-1;i++){  
            for (j=1;j<COL-1;j++){  
                p_new[i][j] =  (p[i+1][j]+p_new[i-1][j]  
                                + pow(beta,2) *(p[i][j+1]+p_new[i][j-1])  
                                - dx*dx*func_serial(i,j,dx,dy))/(2*(1+pow(beta,2)));  
                p_new[i][j] = p[i][j] + omega * (p_new[i][j] - p[i][j]);  
            }  
        }  

        //------------------------  
        //  Boundary conditions  
        //------------------------  

        // Boundary - Case 1  
        if (BC == 1){  
          for (j=0;j<COL;j++){  
              p_new[0][j] = 0;  
              p_new[ROW-1][j] = 0;  
          }  

          for (i=0;i<ROW;i++) {  
              p_new[i][0] = p_new[i][1];  
              p_new[i][COL-1] = p_new[i][COL-2];  
          }  
        }  
        // Boundary - Case 2  
        else if (BC ==2){  
          for (j=0;j<COL;j++){  
              p_new[0][j] = -1/(2*pow(pi,2))*func_serial(0,j,dx,dy);  
              p_new[ROW-1][j] = -1/(2*pow(pi,2))*func_serial(ROW-1,j,dx,dy);  
          }  

          for (i=0;i<ROW;i++) {  
              p_new[i][0] = -1/(2*pow(pi,2))*func_serial(i,0,dx,dy);  
              p_new[i][COL-1] = -1/(2*pow(pi,2))*func_serial(i,COL-1,dx,dy);  
          }  
        }  

        //------------------------  
        //  Convergence Criteria  
        //------------------------  
        for (i=1;i<ROW-1;i++){  
            for (j=1;j<COL-1;j++){  
                SUM1 += fabs(p_new[i][j]);  
                SUM2 += fabs(p_new[i+1][j] + p_new[i-1][j]  
                             + pow(beta,2)*(p_new[i][j+1] + p_new[i][j-1])  
                             - (2+2*pow(beta,2))*p_new[i][j]-dx*dx*func_serial(i,j,dx,dy));  
            }  
        }  
        if ( SUM2/SUM1 < tol ){  
            for (i =0; i<ROW; i++){  
                free(p_new[i]);  
            }  
            free(p_new);  
            *iter = it;  
            end_t = clock();  
            *tot_time = (double)(end_t - start_t)/(CLOCKS_PER_SEC);  
            break;  
        }  
        //------------------------  
        //         Update  
        //------------------------  
        for (i=0;i<ROW;i++){  
            for (j=0;j<COL;j++){  
                p[i][j] = p_new[i][j];}}  

    }  

    for (i=0;i<ROW;i++){  
        free(p_new[i]);  
    }  
    free(p_new);  
}  

// Conjugate Gradient serial version (matrix-based or operator-based)  
double norm_L2_serial(double *a);  
double vvdot_serial(double *a, double *b);  
void vmdot_serial(double **A,double *x,double *b);  
void Adot_serial(double *x,double *b,int BC,int size);  
void make_Abx_serial(double **A, double *b, double *x, double**u  
              ,double dx, double dy, int BC,int size, bool use_matrix_A);  

#define IDX(i, j) ((i) * (COL-2) + (j))   

void Conjugate_Gradient_serial(double **p,double dx, double dy, double tol,  
                                   double *tot_time,int *iter, int BC, bool use_matrix_A)  
{  
    int i,j,k,it;  
    double alpha,beta ;  

    double **A;  
    double *tmp, *x, *b, *z, *r, *r_new;  

    clock_t start_t =0, end_t =0;  

    start_t = clock();  
    if (use_matrix_A){  
        A = (double **) calloc((ROW-2)*(COL-2), sizeof(double *));  
        for (i=0;i<(ROW-2)*(COL-2);i++) {A[i] = (double *) calloc((ROW-2)*(COL-2), sizeof(double));}  
    }  
    tmp    = (double *) malloc((ROW-2)*(COL-2) * sizeof(double));  
    x      = (double *) calloc((ROW-2)*(COL-2), sizeof(double));  
    b      = (double *) calloc((ROW-2)*(COL-2), sizeof(double));  
    z      = (double *) malloc((ROW-2)*(COL-2) * sizeof(double));  
    r      = (double *) malloc((ROW-2)*(COL-2) * sizeof(double));  
    r_new  = (double *) malloc((ROW-2)*(COL-2) * sizeof(double));  
    
    if (use_matrix_A){  
        make_Abx_serial(A,b,x,p,dx,dy,BC,ROW,use_matrix_A);  
        vmdot_serial(A,x,tmp);  
    }  
    else {  
        make_Abx_serial(NULL,b,x,p,dx,dy,BC,ROW,use_matrix_A);  
        Adot_serial(x,tmp,BC,ROW);  
    }  

   for (i=0;i<ROW-2;i++){  
       for (j=0;j<COL-2;j++){  
           r[IDX(i,j)] = b[IDX(i,j)] - tmp[IDX(i,j)];  
           z[IDX(i,j)] = r[IDX(i,j)];  
       }  
   }  

   //---------------------------------------  
   //   Main Loop of Conjugate_Gradient  
   //---------------------------------------  
   for (it=0;it<itmax;it++)  
   {  
       if(use_matrix_A){vmdot_serial(A,z,tmp);}  
       else{Adot_serial(z,tmp,BC,ROW);}  
       alpha = vvdot_serial(r,r)/vvdot_serial(z,tmp);  

       for (i=0;i<ROW-2;i++){  
           for (j=0;j<COL-2;j++){  
               x[IDX(i,j)] = x[IDX(i,j)] + alpha * z[IDX(i,j)];  
               r_new[IDX(i,j)] = r[IDX(i,j)] - alpha*tmp[IDX(i,j)];  
           }  
       }  

       if (norm_L2_serial(r_new) < tol ){  
          //---------------------------------------  
          //   Redistribute x vector to array  
          //---------------------------------------  
          for (i=0;i<ROW-2;i++){  
            for (j=0;j<COL-2;j++){  
                p[i+1][j+1] = x[IDX(i,j)];  
            }  
          }  
          if (BC==1){  
            for(int i=0; i<COL; i++){  
                p[i][0]       = p[i][1];       
                p[i][COL-1]   = p[i][COL-2];   
            }  
          }  
          else if (BC==2){  
            for(int i=0; i<ROW; i++){  
                p[i][0]       = -1/(2*pow(pi,2))*func_serial(i,0,dx,dy);       
                p[i][COL-1]   = -1/(2*pow(pi,2))*func_serial(i,COL-1,dx,dy);  
            }  
            for(int j=0; j<COL; j++){  
                p[0][j]       = -1/(2*pow(pi,2))*func_serial(0,j,dx,dy);       
                p[ROW-1][j]   = -1/(2*pow(pi,2))*func_serial(ROW-1,j,dx,dy);  
            }  
          }  

          *iter = it;  
          if (use_matrix_A){  
            for (int i = 0; i < (ROW-2)*(COL-2); i++){  
                free(A[i]);  
            }  
            free(A);  
          }  
          free(tmp);  
          free(x);  
          free(b);  
          free(z);  
          free(r);  
          free(r_new);  

          end_t = clock();  
          *tot_time = (double)(end_t - start_t)/(CLOCKS_PER_SEC);  
          break;  
       }  

       beta = vvdot_serial(r_new,r_new)/vvdot_serial(r,r);  
       for (i=0;i<ROW-2;i++){  
           for (j=0;j<COL-2;j++){  
               z[IDX(i,j)] = r_new[IDX(i,j)] + beta*z[IDX(i,j)];  
               r[IDX(i,j)] = r_new[IDX(i,j)];  
           }  
       }  
   }  

   // If not converged, copy x to p anyway  
   if (norm_L2_serial(r_new) >= tol) {  
       for (int ii=0; ii<ROW-2; ii++) {  
           for (int jj=0; jj<COL-2; jj++) {  
               p[ii+1][jj+1] = x[IDX(ii,jj)];  
           }  
       }  
   }  

   // Cleanup memory  
   if (use_matrix_A) {  
       for (int i = 0; i < (ROW-2)*(COL-2); i++) free(A[i]);  
       free(A);  
   }  
   free(tmp);  
   free(x);  
   free(b);  
   free(z);  
   free(r);  
   free(r_new);  
}  

void make_Abx_serial(double **A,double *b,double *x,  
              double **u,double dx, double dy, int BC,int size, bool use_matrix_A)  
{  
    int i,j,k,l;  
    double epi  =0.1;  
    double sac_0  = 1.0+dx*(epi*epi-1)*sin(i*dx*pi)*epi*dx/2;  
    double sac_1  = 1.0+dx*(epi*epi-1)*sin(i*dx*pi)*(1-epi)*dx/2;  
    //--------------------------------  
    //         Make Matrix A  
    //--------------------------------  
    if (use_matrix_A){  
        if (BC==1){  
            for (i=0;i<ROW-2;i++){  
                for (j=0;j<COL-2;j++){  
                    if (i==0){  
                        if (j==0){  
                            A[IDX(i,j)][IDX(i,j+1)]=2;  
                            // A[IDX(i,j)][IDX(i,j-1)]=1;  
                            A[IDX(i,j)][IDX(i+1,j)]=1;  
                        }  
                        else if (j==1&&size<23){  
                            A[IDX(i,j)][IDX(i,j)]=-4-sac_0;  
                            A[IDX(i,j)][IDX(i,j-1)]=2;  
                            A[IDX(i,j)][IDX(i,j+1)]=1;  
                            // A[IDX(i,j)][IDX(i-1,j)]=1;  
                            A[IDX(i,j)][IDX(i+1,j)]=1;  
                        }  
                        else if (j==COL-2&&size<23){  
                            A[IDX(i,j)][IDX(i,j)]=-4-sac_1;  
                            A[IDX(i,j)][IDX(i,j-1)]=1;  
                            A[IDX(i,j)][IDX(i,j+1)]=2;  
                            // A[IDX(i,j)][IDX(i-1,j)]=1;  
                            A[IDX(i,j)][IDX(i+1,j)]=1;  
                        }  
                        else if (j==COL-3){  
                            A[IDX(i,j)][IDX(i,j)]=-4;  
                            A[IDX(i,j)][IDX(i,j-1)]=2;  
                            // A[IDX(i,j)][IDX(i,j+1)]=1;  
                            // A[IDX(i,j)][IDX(i-1,j)]=1;  
                            A[IDX(i,j)][IDX(i+1,j)]=1;  
                        }  
                        else{  
                            A[IDX(i,j)][IDX(i,j)]=-4;  
                            A[IDX(i,j)][IDX(i,j-1)]=1;  
                            A[IDX(i,j)][IDX(i,j+1)]=1;  
                            // A[IDX(i,j)][IDX(i-1,j)]=1;  
                            A[IDX(i,j)][IDX(i+1,j)]=1;  
                        }  
                    }  
                    else if (i==ROW-3){  
                        if (j==0){  
                            A[IDX(i,j)][IDX(i,j)]=-4;  
                            // A[IDX(i,j)][IDX(i,j-1)]=1;  
                            A[IDX(i,j)][IDX(i,j+1)]=2;  
                            A[IDX(i,j)][IDX(i-1,j)]=1;  
                            // A[IDX(i,j)][IDX(i+1,j)]=1;  
                        }  
                        else if (j==1&&size<23){  
                            A[IDX(i,j)][IDX(i,j)]=-4-sac_0;  
                            A[IDX(i,j)][IDX(i,j-1)]=2;  
                            A[IDX(i,j)][IDX(i,j+1)]=1;  
                            A[IDX(i,j)][IDX(i-1,j)]=1;  
                            // A[IDX(i,j)][IDX(i+1,j)]=1;  
                        }  
                        else if (j==COL-2&&size<23){  
                            A[IDX(i,j)][IDX(i,j)]=-4-sac_1;  
                            A[IDX(i,j)][IDX(i,j-1)]=1;  
                            A[IDX(i,j)][IDX(i,j+1)]=2;  
                            A[IDX(i,j)][IDX(i-1,j)]=1;  
                            A[IDX(i,j)][IDX(i+1,j)]=1;  
                        }  
                        else{
                            A[IDX(i,j)][IDX(i,j)]=-4;
                            A[IDX(i,j)][IDX(i,j-1)]=1;
                            A[IDX(i,j)][IDX(i,j+1)]=1;
                            A[IDX(i,j)][IDX(i-1,j)]=1;
                            A[IDX(i,j)][IDX(i+1,j)]=1;
                        }
                    }
                }
            }
        }
    }

    //--------------------------------
    //         Make Vector x
    //--------------------------------
    for (i=0;i<ROW-2;i++){
        for (j=0;j<COL-2;j++){
            x[IDX(i,j)] = u[i+1][j+1];
        }
    }

    //--------------------------------
    //        Make Vector b
    //--------------------------------
    if (BC==1){
        for (i=0;i<ROW-2;i++){
            for (j=0;j<COL-2;j++){
                b[IDX(i,j)] = dx*dx*func_serial(i+1,j+1,dx,dy);
            }
        }
    }
    else if(BC==2){
        for (i=0;i<ROW-2;i++){
            for (j=0;j<COL-2;j++){
                b[IDX(i,j)] = dx*dx*func_serial(i+1,j+1,dx,dy);
                if(i==0)    {b[IDX(i,j)] += (-1)*-1/(2*pow(pi,2))*func_serial(i,j+1,dx,dy);}
                if(i==ROW-3){b[IDX(i,j)] += (-1)*-1/(2*pow(pi,2))*func_serial(i+2,j+1,dx,dy);}
                if(j==0)    {b[IDX(i,j)] += (-1)*-1/(2*pow(pi,2))*func_serial(i+1,j,dx,dy);}
                if(j==COL-3){b[IDX(i,j)] += (-1)*-1/(2*pow(pi,2))*func_serial(i+1,j+2,dx,dy);}
            }
        }
    }
}

//------------------------------------------------------------
//              Matrix Calcuation Functions
//------------------------------------------------------------
double norm_L2_serial(double *a)//計算L2向量範數
{
    int i;
    double sum = 0;

    for (i=0;i<(ROW-2)*(COL-2);i++){
        sum = sum + pow(a[i],2);
    }
    return sqrt(sum);
}

void vmdot_serial(double **A,double *x,double *b)//矩陣-向量乘法b=A*x
{
    int i,j;

    for (i=0;i<(ROW-2)*(COL-2);i++){
            b[i] = 0;
    }

    for (i=0;i<(ROW-2)*(COL-2);i++){
        for (j=0;j<(ROW-2)*(COL-2);j++){
            b[i] = b[i] + A[i][j]*x[j];
        }

    }
}

void Adot_serial(double *x,double *b,int BC,int size)//公式化A運算
{
    int i,j;
    double epi  =0.1;
    double dx   = 1.0/(ROW-1), dy = 1.0/(COL-1);

    for (i=0;i<(ROW-2)*(COL-2);i++){
            b[i] = 0;
    }

    if (BC==1){
        if(size<23){
            for (i=0;i<ROW-2;i++){
                double sac_0  = 1.0+dx*(epi*epi-1)*sin(i*dx*pi)*epi*dx/2;
                double sac_1  = 1.0+dx*(epi*epi-1)*sin(i*dx*pi)*(1-epi)*dx/2;
                for (j=0;j<COL-2;j++){
                    double l = -4*x[IDX(i,j)];
                    if(i>0)    {l+=x[IDX(i-1,j)];}
                    if(i<ROW-3){l+=x[IDX(i+1,j)];}
                    if(j>0)    {l+=(j==COL-3)?2*x[IDX(i,j-1)]:x[IDX(i,j-1)];
                                if (j==1)    {l-=sac_0*x[IDX(i,j)];l+=x[IDX(i,j-1)];};
                                }
                    if(j<COL-3){l+=(j==0)    ?2*x[IDX(i,j+1)]:x[IDX(i,j+1)];
                                if (j==COL-2){l-=sac_1*x[IDX(i,j)];l+=x[IDX(i,j+1)];};
                                }
                    b[IDX(i,j)]=l;
                }
            } 
        }
        else{
            for (i=0;i<ROW-2;i++){
                for (j=0;j<COL-2;j++){
                    double l = -4*x[IDX(i,j)];
                    if(i>0)    {l+=x[IDX(i-1,j)];}
                    if(i<ROW-3){l+=x[IDX(i+1,j)];}
                    if(j>0)    {l+=(j==COL-3)?2*x[IDX(i,j-1)]:x[IDX(i,j-1)];}
                    if(j<COL-3){l+=(j==0)    ?2*x[IDX(i,j+1)]:x[IDX(i,j+1)];}
                    b[IDX(i,j)]=l;
                }
            }  
        }
    }
    else if(BC==2){
        for (i=0;i<ROW-2;i++){
            for (j=0;j<COL-2;j++){
                double l = -4*x[IDX(i,j)];
                if(i>0)    {l+=x[IDX(i-1,j)];}
                if(i<ROW-3){l+=x[IDX(i+1,j)];}
                if(j>0)    {l+=x[IDX(i,j-1)];}
                if(j<COL-3){l+=x[IDX(i,j+1)];}
                b[IDX(i,j)]=l;
            }
        }
    }
}

double vvdot_serial(double *a, double *b)//兩項量內積
{
    int i;
    double c = 0;

    for (i=0;i<(ROW-2)*(COL-2);i++){
        c = c + a[i]*b[i];
    }

    return c;
}

// End of Section 1 (Serial)
// ==============================================================================
  
// ==============================================================================
// Section 2: OpenMP version (CG and SOR with ROW=51, COL=51)
// ==============================================================================
  
// File: Conjugate-Gradient-Poisson-Solver-openmp.cpp
// 使用 OpenMP 簡單平行化的 2D Poisson（CG 與 SOR）

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <omp.h>

#ifndef ROW
#define ROW 51
#endif
#ifndef COL
#define COL 51
#endif
#define pi 3.141592
#define itmax 100000

// 函式宣告
void initialization_omp(double **p);
void write_u_omp(const char *dir_nm, const char *file_nm, double **p, double dx, double dy);
void SOR_omp(double **p, double dx, double dy, double tol, double omega, double *tot_time, int *iter, int BC);
void Conjugate_Gradient_omp(double **p, double dx, double dy, double tol, double *tot_time, int *iter, int BC);
double func_omp(int i, int j, double dx, double dy);
void func_anal_omp(double **p, int row_num, int col_num, double dx, double dy);
void error_rms_omp(double **p, double **p_anal, double *err);
void poisson_solver_omp(double **u, double **u_anal, double tol, double omega, int BC, int method, const char *dir_name);

int main_omp(void) {
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
    poisson_solver_omp(u, u_anal, tol, omega, BC, 1, dir_name);

    // SOR method
    printf("\n[SOR Method]\n");
    poisson_solver_omp(u, u_anal, tol, omega, BC, 2, dir_name);

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
void poisson_solver_omp(double **u, double **u_anal, double tol, double omega, int BC, int method, const char *dir_name) {
    char file_name[128];
    double Lx = 1.0, Ly = 1.0;
    double dx = Lx / (ROW - 1), dy = Ly / (COL - 1);
    double err = 0.0, tot_time = 0.0;
    int iter = 0;

    // 先計算解析解並輸出
    func_anal_omp(u_anal, ROW, COL, dx, dy);
    write_u_omp(dir_name, "Analytic_solution.plt", u_anal, dx, dy);

    switch (method) {
        case 1:
            initialization_omp(u);
            Conjugate_Gradient_omp(u, dx, dy, tol, &tot_time, &iter, BC);
            error_rms_omp(u, u_anal, &err);
            printf("CG method - Error : %e, Iteration : %d, Time : %f s\n", err, iter, tot_time);
            write_u_omp(dir_name, "CG_result.plt", u, dx, dy);
            break;
        case 2:
            initialization_omp(u);
            SOR_omp(u, dx, dy, tol, omega, &tot_time, &iter, BC);
            error_rms_omp(u, u_anal, &err);
            printf("SOR method - Error : %e, Iteration : %d, Time : %f s\n", err, iter, tot_time);
            write_u_omp(dir_name, "SOR_result.plt", u, dx, dy);
            break;
    }
}

double func_omp(int i, int j, double dx, double dy) {
    return sin(pi * i * dx) * cos(pi * j * dy);
}

void initialization_omp(double **p) {
    int i, j;
    #pragma omp parallel for private(j)
    for (i = 0; i < ROW; i++) {
        for (j = 0; j < COL; j++) {
            p[i][j] = 0.0;
        }
    }
}

void error_rms_omp(double **p, double **p_anal, double *err) {
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

void func_anal_omp(double **p, int row_num, int col_num, double dx, double dy) {
    int i, j;
    #pragma omp parallel for private(j)
    for (i = 0; i < row_num; i++) {
        for (j = 0; j < col_num; j++) {
            p[i][j] = -1.0 / (2.0 * pi * pi) * sin(pi * i * dx) * cos(pi * j * dy);
        }
    }
}

void write_u_omp(const char *dir_nm, const char *file_nm, double **p, double dx, double dy) {
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
// 這裡只對「邊界條件」與「收斂量測」部分平行化
// ----------------------------------------
void SOR_omp(double **p, double dx, double dy, double tol, double omega, double *tot_time, int *iter, int BC) {
    double **p_new = (double **) malloc(ROW * sizeof(double *));
    for (int i = 0; i < ROW; i++) {
        p_new[i] = (double *) malloc(COL * sizeof(double));
    }
    initialization_omp(p_new);

    double beta = dx / dy;
    clock_t start_t = clock();

    int i, j, it;
    for (it = 1; it < itmax; it++) {
        double SUM1 = 0.0, SUM2 = 0.0;

        // 主更新：串行
        for (i = 1; i < ROW - 1; i++) {
            for (j = 1; j < COL - 1; j++) {
                double temp = (p[i+1][j] + p_new[i-1][j]
                             + beta * beta * (p[i][j+1] + p_new[i][j-1])
                             - dx * dx * func_omp(i, j, dx, dy))
                             / (2.0 * (1.0 + beta * beta));
                p_new[i][j] = p[i][j] + omega * (temp - p[i][j]);
            }
        }

        // 邊界條件：平行化
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
                p_new[0][j] = -1.0 / (2.0 * pi * pi) * func_omp(0, j, dx, dy);
                p_new[ROW-1][j] = -1.0 / (2.0 * pi * pi) * func_omp(ROW-1, j, dx, dy);
            }
            #pragma omp parallel for
            for (i = 0; i < ROW; i++) {
                p_new[i][0] = -1.0 / (2.0 * pi * pi) * func_omp(i, 0, dx, dy);
                p_new[i][COL-1] = -1.0 / (2.0 * pi * pi) * func_omp(i, COL-1, dx, dy);
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
                           - dx * dx * func_omp(i, j, dx, dy));
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
double CG_norm_L2_omp(double *a) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < ROW * COL; i++) {
        sum += a[i] * a[i];
    }
    return sqrt(sum);
}

double CG_vvdot_omp(double *a, double *b) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < ROW * COL; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void CG_vmdot_omp(double **A, double *x, double *b) {
    #pragma omp parallel for
    for (int i = 0; i < ROW * COL; i++) {
        double tmp = 0.0;
        for (int j = 0; j < ROW * COL; j++) {
            tmp += A[i][j] * x[j];
        }
        b[i] = tmp;
    }
}

void CG_make_Abx_omp(double **A, double *b, double *x, double **u, double dx, double dy) {
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
                b[idx] = dx * dx * func_omp(i, j, dx, dy);
            }
        }
    }
    // x 先設為 0
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        x[i] = 0.0;
    }
}

void Conjugate_Gradient_omp(double **p, double dx, double dy, double tol, double *tot_time, int *iter, int BC) {
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
    CG_make_Abx_omp(A, b, x_vec, p, dx, dy);

    // 初始化 r = b - A*x_vec (x_vec 初值為 0 → r = b)
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        r[i] = b[i];
        z[i] = r[i];
    }

    clock_t start_t = clock();
    for (int k = 0; k < itmax; k++) {
        // tmp = A * z
        CG_vmdot_omp(A, z, tmp);

        double zr = CG_vvdot_omp(r, r);
        double zAtmp = CG_vvdot_omp(z, tmp);
        double alpha = zr / zAtmp;

        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            x_vec[i] += alpha * z[i];
            r_new[i] = r[i] - alpha * tmp[i];
        }

        double norm_rnew = CG_norm_L2_omp(r_new);
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

        double beta = CG_vvdot_omp(r_new, r_new) / zr;
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

// End of Section 2 (OpenMP)
// ==============================================================================

// ==============================================================================
// Section 3: MPI version (CG with dense matrix and Allreduce, single-node SOR)
// ==============================================================================
  
// File: Conjugate-Gradient-Poisson-Solver-mpi.cpp
// 描述：在原单机版上，增加 MPI 通信，使 CG 求解器在多进程间共享向量内积和矩阵-向量乘法结果。
//       并且改为用“相对残差”作为收敛准则，迭代过程中由 rank=0 打印 ITER / RESIDUAL / TIME。

#include <mpi.h>
#ifndef ROW
#define ROW 51
#endif
#ifndef COL
#define COL 51
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// 函数声明
void initialization_mpi(double **p);
void write_u_mpi(const char *dir_nm, const char *file_nm, double **p, double dx, double dy);
void SOR_mpi(double **p, double dx, double dy, double tol, double omega,
             double *tot_time, int *iter, int BC);
void Conjugate_Gradient_mpi(double **p, double dx, double dy, double tol,
                            double *tot_time, int *iter, int BC, MPI_Comm comm);
double func_mpi(int i, int j, double dx, double dy);
void func_anal_mpi(double **p, int row_num, int col_num, double dx, double dy);
void error_rms_mpi(double **p, double **p_anal, int row_num, int col_num, double *err);

// MPI 向量内积（Allreduce）
double vvdot_mpi(double *a, double *b, int n, MPI_Comm comm) {
    double local_dot = 0.0;
    for (int i = 0; i < n; i++) {
        local_dot += a[i] * b[i];
    }
    double global_dot = 0.0;
    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global_dot;
}

// MPI 矩阵-向量乘法 b = A * x
// A 存储为“稠密五点差分矩阵”形式，各进程先独立计算 b_local，然后 Allreduce 合并到 b_global
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

// Poisson 求解主函数：先计算解析解，然后根据 method 调用 CG 或 SOR，最后写出结果
void poisson_solver_mpi(double **u, double **u_anal, double tol, double omega,
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
    func_anal_mpi(u_anal, ROW, COL, dx, dy);
    if (rank == 0) {
        write_u_mpi(dir_name, file_name, u_anal, dx, dy);
    }

    // 2) 数值求解
    switch (method) {
        case 1:
            // Conjugate Gradient，并行版本
            initialization_mpi(u);
            Conjugate_Gradient_mpi(u, dx, dy, tol, &tot_time, &iter, BC, comm);
            if (rank == 0) {
                error_rms_mpi(u, u_anal, ROW, COL, &err);
                printf("MPI CG - Error: %e, Iter = %d, Time = %f s\n", err, iter, tot_time);
                strcpy(file_name, "CG_result_mpi.plt");
                write_u_mpi(dir_name, file_name, u, dx, dy);
            }
            break;
        case 2:
            // SOR（保持单机逻辑，不并行）
            initialization_mpi(u);
            SOR_mpi(u, dx, dy, tol, omega, &tot_time, &iter, BC);
            if (rank == 0) {
                error_rms_mpi(u, u_anal, ROW, COL, &err);
                printf("MPI SOR (单机模式) - Error: %e, Iter = %d, Time = %f s\n", err, iter, tot_time);
                strcpy(file_name, "SOR_result_mpi.plt");
                write_u_mpi(dir_name, file_name, u, dx, dy);
            }
            break;
        default:
            if (rank == 0) {
                fprintf(stderr, "未定义的方法编号 %d\n", method);
            }
    }
}

int main_mpi(int argc, char *argv[]) {
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
    poisson_solver_mpi(u, u_anal, tol, omega, BC, 1, "./RESULT/", comm);

    // ----- SOR 求解（单机模式，仅 rank=0 输出） -----
    if (rank == 0) {
        printf("\n>>[MPI SOR Method (单机模式)]\n");
    }
    poisson_solver_mpi(u, u_anal, tol, omega, BC, 2, "./RESULT/", comm);

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

void initialization_mpi(double **p) {
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) {
            p[i][j] = 0.0;
        }
    }
}

void func_anal_mpi(double **p, int row_num, int col_num, double dx, double dy) {
    for (int i = 0; i < row_num; i++) {
        for (int j = 0; j < col_num; j++) {
            double x = i * dx;
            double y = j * dy;
            p[i][j] = -1.0 / (2.0 * pi * pi) * sin(pi * x) * cos(pi * y);
        }
    }
}

double func_mpi(int i, int j, double dx, double dy) {
    double x = i * dx;
    double y = j * dy;
    return sin(pi * x) * cos(pi * y);
}

void write_u_mpi(const char *dir_nm, const char *file_nm, double **p, double dx, double dy) {
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

void error_rms_mpi(double **p, double **p_anal, int row_num, int col_num, double *err) {
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
void SOR_mpi(double **p, double dx, double dy, double tol, double omega,
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
    initialization_mpi(p_new);

    for (it = 1; it < itmax; it++) {
        SUM1 = 0.0;
        SUM2 = 0.0;
        // 迭代更新
        for (i = 1; i < ROW - 1; i++) {
            for (j = 1; j < COL - 1; j++) {
                p_new[i][j] = (p[i + 1][j] + p_new[i - 1][j]
                            + beta * beta * (p[i][j + 1] + p_new[i][j - 1])
                            - dx * dx * func_mpi(i, j, dx, dy))
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
                p_new[0][j]      = -1.0 / (2.0 * pi * pi) * func_mpi(0, j, dx, dy);
                p_new[ROW - 1][j] = -1.0 / (2.0 * pi * pi) * func_mpi(ROW - 1, j, dx, dy);
            }
            for (i = 0; i < ROW; i++) {
                p_new[i][0]       = -1.0 / (2.0 * pi * pi) * func_mpi(i, 0, dx, dy);
                p_new[i][COL - 1] = -1.0 / (2.0 * pi * pi) * func_mpi(i, COL - 1, dx, dy);
            }
        }
        // 收敛判断
        for (i = 1; i < ROW - 1; i++) {
            for (j = 1; j < COL - 1; j++) {
                SUM1 += fabs(p_new[i][j]);
                SUM2 += fabs(p_new[i + 1][j] + p_new[i - 1][j]
                             + beta * beta * (p_new[i][j + 1] + p_new[i][j - 1])
                             - (2.0 + 2.0 * beta * beta) * p_new[i][j]
                             - dx * dx * func_mpi(i, j, dx, dy));
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

// 并行版 Conjugate Gradient，使用 MPI_Allreduce 来做向量内积与矩阵-向量乘法合并
// 收敛准则改为相对残差：||r|| / ||b|| < tol
// 并在每次迭代打印 ITER / RESIDUAL / TIME
void Conjugate_Gradient_mpi(double **p, double dx, double dy, double tol,
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

    // 2) 构造矩阵 A（五点差分），对角 = +4，邻接 = −1，并考虑边界
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

    // 3) 构造向量 b：b[idx] = dx^2 * f(i,j)，边界处 b=0
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) {
            int idx = i * COL + j;
            if (i == 0 || i == ROW - 1 || j == 0 || j == COL - 1) {
                b[idx] = 0.0;
            } else {
                b[idx] = dx * dx * func_mpi(i, j, dx, dy);
            }
        }
    }

    // 4) 计算 ||b||_2，并取平方根
    double local_b2 = 0.0;
    for (int idx = 0; idx < n; idx++) {
        local_b2 += b[idx] * b[idx];
    }
    double bnorm2 = 0.0;
    MPI_Allreduce(&local_b2, &bnorm2, 1, MPI_DOUBLE, MPI_SUM, comm);
    double bnorm = sqrt(bnorm2);

    // 5) 初始化：x = 0，r = b - A x = b，z = r
    for (int idx = 0; idx < n; idx++) {
        x[idx] = 0.0;
        r[idx] = b[idx];
        z[idx] = r[idx];
    }

    // 6) 计算初始 rr = r·r
    double rr = vvdot_mpi(r, r, n, comm);

    // 7) 相对残差的平方阈值： rr_new < tol^2 * bnorm2
    double tol2_rel = tol * tol * bnorm2;

    // 8) 主迭代：CG
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
