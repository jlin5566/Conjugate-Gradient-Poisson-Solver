// no parallelization, no even-odd ordering, assign N, method, BC, use_matrix_A in the command
#define ROW 100
#define COL 100
#define pi 3.141592
#define itmax 1000000

#include <stdio.h>
#include <math.h>

// #include <mpi.h>
// #include <omp.h>
#include <string.h>


void initialization(double **p);
void write_u(const char *dir_nm,const char *file_nm, double **p,double dx, double dy);

//-----------------------------------
// Poisson Solvers - CG, SOR, CG
//-----------------------------------
void SOR(double **p,double dx, double dy, double tol, double omega,
                               double *tot_time,int *iter,int BC);
void Conjugate_Gradient(double **p,double dx, double dy, double tol,
                                   double *tot_time,int *iter,int BC,bool use_matrix_A);

//-----------------------------------
//        Mathematical tools
//-----------------------------------
double func(int i, int j, double dx, double dy);
void func_anal(double **p, int row_num, int col_num, double dx, double dy);
void error_rms(double **p, double **p_anal, double *err);
void poisson_solver(double **u, double **u_anal, double tol, double omega,
                    int BC, int method, const char *dir_name,bool use_matrix_A){

  const char *file_name ;

  int iter = 0;
  double Lx = 1.0, Ly = 1.0;
  double dx, dy, err = 0, tot_time = 0;

  dx = Lx/(ROW-1);
  dy = Ly/(COL-1);

  //-----------------------------
  //      Analytic Solutions
  //-----------------------------
  file_name = "Analytic_solution.plt";
  func_anal(u_anal,ROW,COL,dx,dy);
  write_u(dir_name,file_name,u_anal,dx,dy);

  switch (method) {
    case 1 :
       //-----------------------------
       //  Conjugate Gradient Method
       //-----------------------------
       initialization(u);
       Conjugate_Gradient(u,dx,dy,tol,&tot_time,&iter,BC,use_matrix_A);
       error_rms(u,u_anal,&err);
       printf("CG_%d method - Error : %e, Iteration : %d, Time : %f s \n",BC,err,iter,tot_time);
       if (BC==1){file_name = "CG_result_1.plt";}
       else if (BC==2){file_name = "CG_result_2.plt";}
       write_u(dir_name,file_name,u,dx,dy);
      break;

    case 2 :
       //-----------------------------
       //         SOR Method
       //-----------------------------
       initialization(u);
       SOR(u,dx,dy,tol,omega,&tot_time,&iter,BC);
       error_rms(u,u_anal,&err);
       printf("SOR_%d Method - Error : %e, Iteration : %d, Time : %f s \n",BC,err,iter,tot_time);
       if (BC==1){file_name = "SOR_result_1.plt";}
       else if (BC==2){file_name = "SOR_result_2.plt";}
       write_u(dir_name,file_name,u,dx,dy);
      break;
  }

}

double func(int i,int j,double dx,double dy)
{
    return sin(pi*i*dx)*cos(pi*j*dy);
}

void initialization(double **p)
{
    int i,j;
    for (i=0;i<ROW;i++){
        for (j=0;j<COL;j++){
            p[i][j] = 0; }}

}

void error_rms(double **p, double **p_anal, double *err)
{
  int i,j;
  for (i=0;i<ROW;i++){
    for (j=0;j<COL;j++){
      *err = *err + pow(p[i][j] -p_anal[i][j],2);
    }
  }

  *err = sqrt(*err)/(ROW*COL);
}


void func_anal(double **p, int row_num, int col_num, double dx, double dy)
{
    int i,j;
    for (i=0;i<row_num;i++){
        for (j=0;j<col_num;j++){
            p[i][j] = -1/(2*pow(pi,2))*sin(pi*i*dx)*cos(pi*j*dy); }}
}

void write_u(const char *dir_nm,const char *file_nm, double **p,double dx, double dy)
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

// -------- File: Poisson_Equation.c --------
//
//
//  Programming for 2D Poisson Equation solving 2D Poisson Equation
//
//  Laplace(u(x,y)) = f(x,y) for (x,y) in domain
//  on the boundary boundary_D and boundary_N
//
//  u(x,y) = g(x,y) on boundary_D
//  du/dn  = h(x,y) on boundary N
//
//
//  Domain
//      [x,y] is in [0,1] X [0,1]
//      f(x,y) = sin(pi*x) * cos(pi *y)
//      Analytic solution is
//          u^a(x,y) = -1/(2*pi^2) * sin(pi*x) * cos(pi*y)
//
//  Boundary Condition
//      Case 1
//          Dirichlet bondary condition u(x,y) = 0 in y axis
//          Neumann boundary condition du/dn = 0 in x axis
//
//      Case 2
//          Dirichlet boundary conditioins using analytical solution
//          both x and y directions.
//

#include <stdlib.h>

//-----------------------------------
//        Productivity tools
//-----------------------------------

void poisson_solver(double **u, double **u_anal, double tol, double omega,
                    int BC, int method, const char *dir_name,bool use_matrix_A);

int main(int argc, char *argv[])
{
    int method   = 1;
    int BC       = 1;
    bool use_matrix_A = false;
    if (argc==3){
        method   = atoi(argv[1]);
        BC       = atoi(argv[2]);
    }
    else if (argc ==4 && strcmp(argv[3], "true") == 0) {use_matrix_A = true;}
    else {printf("Use default setting.\n");}
    double **u;
    double **u_anal;

    const char *dir_name ;

    int i;
    double tol, omega;

    int make_fold= system("mkdir RESULT");

    // --------------------------------------------------------
    //                    Memory allocation
    // --------------------------------------------------------
    u      = (double **) malloc(ROW *sizeof(double *));
    u_anal = (double **) malloc(ROW *sizeof(double *));

    // if (u == NULL || u_anal == NULL){
    //     fprintf(stderr, "malloc error\n");
    //     exit(1);
    // }

    for (i=0;i<ROW;i++)
    {
      u[i]      = (double *) malloc(COL * sizeof(double));
      u_anal[i] = (double *) malloc(COL * sizeof(double));
    //   if (u[i] == NULL || u_anal[i] == NULL){
    //     fprintf(stderr, "malloc col[%d] error\n", i);
    //     exit(1);
    //   }
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

    //----------------------------------------
    //       Poisson Solver Type
    //
    // BC = 1 : Boundary condition Case 1
    //    = 2 : Boundary condition Case 2
    //
    // method = 1 : Conjugate Gradient method
    //        = 2 : SOR method
    //
    //----------------------------------------

    poisson_solver(u,u_anal,tol,omega,BC,method,dir_name,use_matrix_A);

    for (int i = 0; i < ROW; i++){
        free(u[i]);
        free(u_anal[i]);
    }
    free(u);
    free(u_anal);

    return 0;
}

#include <time.h>

void SOR(double **p,double dx, double dy, double tol, double omega,
                               double *tot_time,int *iter,int BC)
{
    int i,j,k,it;
    double beta,rms;
    double SUM1,SUM2;
    double **p_new;
    clock_t start_t =0, end_t =0;

    start_t = clock();
    beta = dx/dy;

    p_new = (double **) malloc(ROW *sizeof(double *));
    for (i=0;i<ROW;i++)
    {
      p_new[i]      = (double *) malloc(COL * sizeof(double));
    }
    initialization(p_new);

    for (it=1;it<itmax;it++){
        SUM1 = 0;
        SUM2 = 0;


        for (i=1;i<ROW-1;i++){
            for (j=1;j<COL-1;j++){
                p_new[i][j] =  (p[i+1][j]+p_new[i-1][j]
                                + pow(beta,2) *(p[i][j+1]+p_new[i][j-1])
                                - dx*dx*func(i,j,dx,dy))/(2*(1+pow(beta,2)));
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
              p_new[0][j] = -1/(2*pow(pi,2))*func(0,j,dx,dy);
              p_new[ROW-1][j] = -1/(2*pow(pi,2))*func(ROW-1,j,dx,dy);
          }

          for (i=0;i<ROW;i++) {
              p_new[i][0] = -1/(2*pow(pi,2))*func(i,0,dx,dy);
              p_new[i][COL-1] = -1/(2*pow(pi,2))*func(i,COL-1,dx,dy);
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
                             - (2+2*pow(beta,2))*p_new[i][j]-dx*dx*func(i,j,dx,dy));
            }
        }
        // if (it % 200 == 0) {
        //     printf("Iteration : %d, SUM1 : %f, SUM2 : %f, Ratio : %f \n",it,SUM1,SUM2,SUM2/SUM1);
        //     }
        if ( SUM2/SUM1 < tol ){
            for (int i =0; i<ROW; i++){
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
}

// -------- File: Conjugate_Gradient.c --------

//-----------------------------------
//      Matrix Calculations
//-----------------------------------
double norm_L2(double *a);
double vvdot(double *a, double *b);
void vmdot(double **A,double *x,double *b);
void Adot(double *x,double *b,int BC,int size);
void make_Abx(double **A, double *b, double *x, double**u
              ,double dx, double dy, int BC,int size, bool use_matrix_A);

//-----------------------------------
//      Mathematical functions
//-----------------------------------

#define IDX(i, j) ((i) * (COL-2) + (j)) 

void Conjugate_Gradient(double **p,double dx, double dy, double tol,
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
        make_Abx(A,b,x,p,dx,dy,BC,ROW,use_matrix_A);
        vmdot(A,x,tmp);
    }
    else {
        make_Abx(NULL,b,x,p,dx,dy,BC,ROW,use_matrix_A);
        Adot(x,tmp,BC,ROW);
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
       if(use_matrix_A){vmdot(A,z,tmp);}
       else{Adot(z,tmp,BC,ROW);}
       alpha = vvdot(r,r)/vvdot(z,tmp);


       for (i=0;i<ROW-2;i++){
           for (j=0;j<COL-2;j++){
               x[IDX(i,j)] = x[IDX(i,j)] + alpha * z[IDX(i,j)];
               r_new[IDX(i,j)] = r[IDX(i,j)] - alpha*tmp[IDX(i,j)];
           }
       }

    //    if (it % 50 == 0) {
    //     printf("iteration=%d, res=%.3e\n", it, norm_L2(r_new));
    //     }

       if (norm_L2(r_new) < tol ){
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
                p[i][0]       = -1/(2*pow(pi,2))*func(i,0,dx,dy);       
                p[i][COL-1]   = -1/(2*pow(pi,2))*func(i,COL-1,dx,dy);  
            }
            for(int j=0; j<COL; j++){
                p[0][j]       = -1/(2*pow(pi,2))*func(0,j,dx,dy);       
                p[ROW-1][j]   = -1/(2*pow(pi,2))*func(ROW-1,j,dx,dy);  
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

       beta = vvdot(r_new,r_new)/vvdot(r,r);
       for (i=0;i<ROW-2;i++){
           for (j=0;j<COL-2;j++){
               z[IDX(i,j)] = r_new[IDX(i,j)] + beta*z[IDX(i,j)];
               r[IDX(i,j)] = r_new[IDX(i,j)];
           }
       }
   }
}

//------------------------------------------------------------
//             Make Stiffness matrix of CG method
//------------------------------------------------------------
void make_Abx(double **A,double *b,double *x,
              double **u,double dx, double dy, int BC, int size, bool use_matrix_A)
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
                            A[IDX(i,j)][IDX(i,j)]=-4;
                            // A[IDX(i,j)][IDX(i,j-1)]=1;
                            A[IDX(i,j)][IDX(i,j+1)]=2;
                            // A[IDX(i,j)][IDX(i-1,j)]=1;
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
                            A[IDX(i,j)][IDX(i,j)]=-4-sac_0;
                            A[IDX(i,j)][IDX(i,j-1)]=1;
                            A[IDX(i,j)][IDX(i,j+1)]=2;
                            A[IDX(i,j)][IDX(i-1,j)]=1;
                            // A[IDX(i,j)][IDX(i+1,j)]=1;
                        }
                        else if (j==COL-3){
                            A[IDX(i,j)][IDX(i,j)]=-4;
                            A[IDX(i,j)][IDX(i,j-1)]=2;
                            // A[IDX(i,j)][IDX(i,j+1)]=1;
                            A[IDX(i,j)][IDX(i-1,j)]=1;
                            // A[IDX(i,j)][IDX(i+1,j)]=1;
                        }
                        else{
                            A[IDX(i,j)][IDX(i,j)]=-4;
                            A[IDX(i,j)][IDX(i,j-1)]=1;
                            A[IDX(i,j)][IDX(i,j+1)]=1;
                            A[IDX(i,j)][IDX(i-1,j)]=1;
                            // A[IDX(i,j)][IDX(i+1,j)]=1;
                        }
                    }
                    else if (i!=0&&i!=ROW-3&&j==0){
                        A[IDX(i,j)][IDX(i,j)]=-4;
                        // A[IDX(i,j)][IDX(i,j-1)]=1;
                        A[IDX(i,j)][IDX(i,j+1)]=2;
                        A[IDX(i,j)][IDX(i-1,j)]=1;
                        A[IDX(i,j)][IDX(i+1,j)]=1;
                    }
                    else if (i!=0&&i!=ROW-3&&j==1&&size<23){
                        A[IDX(i,j)][IDX(i,j)]=-4-sac_0;
                        A[IDX(i,j)][IDX(i,j-1)]=2;
                        A[IDX(i,j)][IDX(i,j+1)]=1;
                        A[IDX(i,j)][IDX(i-1,j)]=1;
                        A[IDX(i,j)][IDX(i+1,j)]=1;
                    }
                    else if (i!=0&&i!=ROW-3&&j==COL-2&&size<23){
                        A[IDX(i,j)][IDX(i,j)]=-4-sac_0;
                        A[IDX(i,j)][IDX(i,j-1)]=1;
                        A[IDX(i,j)][IDX(i,j+1)]=2;
                        A[IDX(i,j)][IDX(i-1,j)]=1;
                        A[IDX(i,j)][IDX(i+1,j)]=1;
                    }
                    else if (i!=0&&i!=ROW-3&&j==COL-3){
                        A[IDX(i,j)][IDX(i,j)]=-4;
                        A[IDX(i,j)][IDX(i,j-1)]=2;
                        // A[IDX(i,j)][IDX(i,j+1)]=1;
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
        else if (BC==2){
            for (i=0;i<ROW-2;i++){
                for (j=0;j<COL-2;j++){
                    if (i==0){
                        if (j==0){
                            A[IDX(i,j)][IDX(i,j)]=-4;
                            // A[IDX(i,j)][IDX(i,j-1)]=1;
                            A[IDX(i,j)][IDX(i,j+1)]=1;
                            // A[IDX(i,j)][IDX(i-1,j)]=1;
                            A[IDX(i,j)][IDX(i+1,j)]=1;
                        }
                        else if (j==COL-3){
                            A[IDX(i,j)][IDX(i,j)]=-4;
                            A[IDX(i,j)][IDX(i,j-1)]=1;
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
                            A[IDX(i,j)][IDX(i,j+1)]=1;
                            A[IDX(i,j)][IDX(i-1,j)]=1;
                            // A[IDX(i,j)][IDX(i+1,j)]=1;
                        }
                        else if (j==COL-3){
                            A[IDX(i,j)][IDX(i,j)]=-4;
                            A[IDX(i,j)][IDX(i,j-1)]=1;
                            // A[IDX(i,j)][IDX(i,j+1)]=1;
                            A[IDX(i,j)][IDX(i-1,j)]=1;
                            // A[IDX(i,j)][IDX(i+1,j)]=1;
                        }
                        else{
                            A[IDX(i,j)][IDX(i,j)]=-4;
                            A[IDX(i,j)][IDX(i,j-1)]=1;
                            A[IDX(i,j)][IDX(i,j+1)]=1;
                            A[IDX(i,j)][IDX(i-1,j)]=1;
                            // A[IDX(i,j)][IDX(i+1,j)]=1;
                        }
                    }
                    else if (i!=0&&i!=ROW-3&&j==0){
                        A[IDX(i,j)][IDX(i,j)]=-4;
                        // A[IDX(i,j)][IDX(i,j-1)]=1;
                        A[IDX(i,j)][IDX(i,j+1)]=1;
                        A[IDX(i,j)][IDX(i-1,j)]=1;
                        A[IDX(i,j)][IDX(i+1,j)]=1;
                    }
                    else if (i!=0&&i!=ROW-3&&j==COL-3){
                        A[IDX(i,j)][IDX(i,j)]=-4;
                        A[IDX(i,j)][IDX(i,j-1)]=1;
                        // A[IDX(i,j)][IDX(i,j+1)]=1;
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
                b[IDX(i,j)] = dx*dx*func(i+1,j+1,dx,dy);
            }
        }
    }
    else if(BC==2){
        for (i=0;i<ROW-2;i++){
            for (j=0;j<COL-2;j++){
                b[IDX(i,j)] = dx*dx*func(i+1,j+1,dx,dy);
                if(i==0)    {b[IDX(i,j)] += (-1)*-1/(2*pow(pi,2))*func(i,j+1,dx,dy);}
                if(i==ROW-3){b[IDX(i,j)] += (-1)*-1/(2*pow(pi,2))*func(i+2,j+1,dx,dy);}
                if(j==0)    {b[IDX(i,j)] += (-1)*-1/(2*pow(pi,2))*func(i+1,j,dx,dy);}
                if(j==COL-3){b[IDX(i,j)] += (-1)*-1/(2*pow(pi,2))*func(i+1,j+2,dx,dy);}
            }
        }
    }
}

//------------------------------------------------------------
//              Matrix Calcuation Functions
//------------------------------------------------------------
double norm_L2(double *a)
{
    int i;
    double sum = 0;

    for (i=0;i<(ROW-2)*(COL-2);i++){
        sum = sum + pow(a[i],2);
    }
    return sqrt(sum);
}

void vmdot(double **A,double *x,double *b)
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

void Adot(double *x,double *b,int BC,int size)
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

double vvdot(double *a, double *b)
{
    int i;
    double c = 0;

    for (i=0;i<(ROW-2)*(COL-2);i++){
        c = c + a[i]*b[i];
    }

    return c;
}
