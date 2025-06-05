#define ROW 51
#define COL 51
#define pi 3.141592
#define itmax 100000

#include <stdio.h>
#include <math.h>

#include <string.h>

void initialization(double **p);
void write_u(char *dir_nm,char *file_nm, double **p,double dx, double dy);

//-----------------------------------
// Poisson Solvers - CG, SOR, CG
//-----------------------------------
void SOR(double **p,double dx, double dy, double tol, double omega,
                               double *tot_time,int *iter,int BC);
void Conjugate_Gradient(double **p,double dx, double dy, double tol,
                                   double *tot_time,int *iter,int BC);

//-----------------------------------
//        Mathematical tools
//-----------------------------------
double func(int i, int j, double dx, double dy);//定義RHS的解
void func_anal(double **p, int row_num, int col_num, double dx, double dy);//定義解析解
void error_rms(double **p, double **p_anal, double *err);//計算數值解與解析解的誤差
void poisson_solver(double **u, double **u_anal, double tol, double omega,
                    int BC, int method, char *dir_name){

  char *file_name ;//輸出的檔案名稱

  int iter = 0;//迭代次數
  double Lx = 1.0, Ly = 1.0;
  double dx, dy, err = 0, tot_time = 0;

  dx = Lx/(ROW-1);
  dy = Ly/(COL-1);

  //-----------------------------
  //      Analytic Solutions
  //-----------------------------
  file_name = "Analytic_solution.plt";//建立解析解並輸出的檔名
  func_anal(u_anal,ROW,COL,dx,dy);//呼叫並計算每一格的解析解
  write_u(dir_name,file_name,u_anal,dx,dy);//把解析解輸出成.plt

  switch (method) {//根據method值選擇要使用的方法
    case 1 :
       //-----------------------------
       //  Conjugate Gradient Method
       //-----------------------------
       initialization(u);//清空矩陣
       Conjugate_Gradient(u,dx,dy,tol,&tot_time,&iter,BC);
       error_rms(u,u_anal,&err);
       printf("CG method - Error : %e, Iteration : %d, Time : %f s \n",err,iter,tot_time);

       file_name = "CG_result.plt";
       write_u(dir_name,file_name,u,dx,dy);
      break;

    case 2 :
       //-----------------------------
       //         SOR Method
       //-----------------------------
       initialization(u);
       SOR(u,dx,dy,tol,omega,&tot_time,&iter,BC);
       error_rms(u,u_anal,&err);
       printf("SOR Method - Error : %e, Iteration : %d, Time : %f s \n",err,iter,tot_time);

       file_name = "SOR_result.plt";
       write_u(dir_name,file_name,u,dx,dy);
      break;
  }

}

double func(int i,int j,double dx,double dy)//定義RHS解
{
    return sin(pi*i*dx)*cos(pi*j*dy);
}

void initialization(double **p)//清空矩陣
{
    int i,j;
    for (i=0;i<ROW;i++){
        for (j=0;j<COL;j++){
            p[i][j] = 0; }}

}

void error_rms(double **p, double **p_anal, double *err)//計算誤差
{
  int i,j;
  for (i=0;i<ROW;i++){
    for (j=0;j<COL;j++){
      *err = *err + pow(p[i][j] -p_anal[i][j],2);
    }
  }

  *err = sqrt(*err)/(ROW*COL);
}


void func_anal(double **p, int row_num, int col_num, double dx, double dy)//精確解
{
    int i,j;
    for (i=0;i<row_num;i++){
        for (j=0;j<col_num;j++){
            p[i][j] = -1/(2*pow(pi,2))*sin(pi*i*dx)*cos(pi*j*dy); }}
}

void write_u(char *dir_nm,char *file_nm, double **p,double dx, double dy)//將結果輸出成檔案
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
//  Laplce(u(x,y)) = f(x,y) for (x,y) in domain
//  on the boundary boundary_D and boundary_N
//
//  u(x,y) = g(x,y) on boundary_D
//  du/dn  = h(x,y) on boundary N
//
//
//  Domian
//      [x,y] is in [0,1] X [0,1]
//      f(x,y) = sin(pi*x) * cos(pi *y)
//      Analytic solution is
//          u^a(x,y) = -1/(2*pi^2) * sin(pi*x) * cos(pi*y)
//
//  Boundary Condition
//      Case 1
//          Diriclet bondary condition u(x,y) = 0 in x direction q
//          Neumann boundary condition du/dn = 0 in y direction
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
                    int BC, int method, char *dir_name);

int main(void)
{
    double **u;
    double **u_anal;

    char *dir_name ;

    int i, method, BC;
    double tol, omega;

    int make_fold= system("mkdir RESULT");

    // --------------------------------------------------------
    //                    Memory allocation
    // --------------------------------------------------------
    u      = (double **) malloc(ROW *sizeof(double));//用來存儲數值解
    u_anal = (double **) malloc(ROW *sizeof(double));//用來存儲解析解

    for (i=0;i<ROW;i++)
    {
      u[i]      = (double *) malloc(COL * sizeof(double));
      u_anal[i] = (double *) malloc(COL * sizeof(double));
    }

    //--------------------
    //   Initial setting
    //--------------------
    tol = 1e-6;
    omega = 1.8;
    dir_name = "./RESULT/";

    printf("\n");
    printf("---------------------------------------- \n");
    printf("Nx : %d, Ny : %d\n",ROW,COL);
    printf("Tolerance : %f, Omega : %f \n",tol, omega);
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
    BC = 1;
    // -----------------------
    //       CG method
    // -----------------------
    printf("\n[CG Method]\n");
    poisson_solver(u, u_anal, tol, omega, BC, 1, dir_name);  // method = 1

    // -----------------------
    //       SOR method
    // -----------------------
    printf("\n[SOR Method]\n");
    poisson_solver(u, u_anal, tol, omega, BC, 2, dir_name);  // method = 2

    // 釋放記憶體

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
    time_t start_t =0, end_t =0;//時間歸零

    start_t = clock();//計時起點
    beta = dx/dy;

    p_new = (double **) malloc(ROW *sizeof(double));
    for (i=0;i<ROW;i++)
    {
      p_new[i]      = (double *) malloc(COL * sizeof(double));
    }
    initialization(p_new);//將矩陣歸零
//主要迴圈
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
        //判斷迴圈是否收斂
        for (i=1;i<ROW-1;i++){
            for (j=1;j<COL-1;j++){
                SUM1 += fabs(p_new[i][j]);
                SUM2 += fabs(p_new[i+1][j] + p_new[i-1][j]
                             + pow(beta,2)*(p_new[i][j+1] + p_new[i][j-1])
                             - (2+2*pow(beta,2))*p_new[i][j]-dx*dx*func(i,j,dx,dy));
            }
        }

        if ( SUM2/SUM1 < tol ){
            free(p_new);
            *iter = it;
            end_t = clock();
            *tot_time = (double)(end_t - start_t)/(CLOCKS_PER_SEC);
            break;
        }
        // printf("Iteration : %d, SUM1 : %f, SUM2 : %f, Ratio : %f \n",it,SUM1,SUM2,SUM2/SUM1);
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

void make_Abx(double **A, double *b, double *x, double**u
              ,double dx, double dy);

//-----------------------------------
//      Mathematical functions
//-----------------------------------

void Conjugate_Gradient(double **p,double dx, double dy, double tol,
                                   double *tot_time,int *iter, int BC)
{
    int i,j,k,it;
    double alpha,beta ;

    double **A;
    double *tmp, *x, *b, *z, *r, *r_new;

    time_t start_t =0, end_t =0;

    start_t = clock();

    A = (double **) malloc(ROW*COL * sizeof(double));
    for (i=0;i<ROW*COL;i++)
    {
      A[i] = (double *) malloc(ROW*COL * sizeof(double));
    }
    tmp    = (double *) malloc(ROW*COL * sizeof(double));
    x      = (double *) malloc(ROW*COL * sizeof(double));
    b      = (double *) malloc(ROW*COL * sizeof(double));
    z      = (double *) malloc(ROW*COL * sizeof(double));
    r      = (double *) malloc(ROW*COL * sizeof(double));
    r_new  = (double *) malloc(ROW*COL * sizeof(double));

    // for (i=0;i<ROW*COL;i++){
    //     for (j=0;j<ROW*COL;j++){
    //         A[i][j] = 0;
    //     }
    // }

    make_Abx(A,b,x,p,dx,dy);
    vmdot(A,x,tmp);

   for (i=0;i<ROW;i++){
       for (j=0;j<COL;j++){
           r[COL*i+j] = b[COL*i+j] - tmp[COL*i+j];
           z[COL*i+j] = r[COL*i+j];
       }
   }

   //---------------------------------------
   //   Main Loop of Conjugate_Gradient
   //---------------------------------------
   for (it=0;it<itmax;it++)
   {
       vmdot(A,z,tmp);
       alpha = vvdot(r,r)/vvdot(z,tmp);


       for (i=0;i<ROW;i++){
           for (j=0;j<COL;j++){
               x[COL*i+j] = x[COL*i+j] + alpha * z[COL*i+j];
               r_new[COL*i+j] = r[COL*i+j] - alpha*tmp[COL*i+j];
           }
       }

       if (norm_L2(r_new) < tol ){
          // printf("iteration : %d, tol : %f, value : %f\n",it,tol,norm_L2(r_new) );
          //---------------------------------------
          //   Redistribute x vector to array
          //---------------------------------------
          for (i=0;i<ROW;i++)
          {
            for (j=0;j<COL;j++)
            {
              p[i][j] = x[COL*i+j];
            }
          }
          *iter = it;
          free(A);
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
       for (i=0;i<ROW;i++){
           for (j=0;j<COL;j++){
               z[COL*i+j] = r_new[COL*i+j] + beta*z[COL*i+j];
               r[COL*i+j] = r_new[COL*i+j];
           }
       }
   }

}

//------------------------------------------------------------
//             Make Stiffness matrix of CG method
//------------------------------------------------------------
void make_Abx(double **A,double *b,double *x,
              double **u,double dx, double dy)
{
    int i,j,k,l;
    //--------------------------------
    //         Make Matrix A
    //--------------------------------
    for (k=0;k<ROW;k++){
        for (l=0;l<COL;l++){
            if (k==l){
                if (k==0 || k==ROW-1){
                  for (i=0;i<ROW;i++){
                      A[COL*k+i][ROW*l+i]   = 1;
                  }
                }
                else{
                  for (i=0;i<ROW;i++){
                      if (i == 0){
                          A[COL*k+i][ROW*l+i]   = -1;
                          A[COL*k+i+1][ROW*l+i] = 1;
                      }
                      else if (i == ROW-1){
                          A[COL*k+i][ROW*l+i]   = -1;
                          A[COL*k+i-1][ROW*l+i] = 1;
                      }
                      else {
                          A[COL*k+i][ROW*l+i]   = -4;
                          A[COL*k+i-1][ROW*l+i] = 1;
                          A[COL*k+i+1][ROW*l+i] = 1;
                      }
                  }
                }
            }

            else if ( abs(k-l) == 1 && k!=0 && k!=ROW-1){
                for (i=0;i<ROW;i++){
                  if (i==0 || i==ROW-1)
                    A[COL*k+i][ROW*l+i] = 0;
                  else
                    A[COL*k+i][ROW*l+i] = 1;
                }
            }
            else{
                for (i=0;i<ROW;i++){
                    for (j=0;j<COL;j++){
                        A[COL*k+i][ROW*l+j] = 0;
                    }
                }
            }
            // printf("i: %d, j :  %d \n",k,l);
            // for (j=0;j<ROW;j++){
            //   printf("%d ",i);
            //   for (i=0;i<COL;i++){
            //       printf("%f ",A[COL*k+i][ROW*l+j]);
            //   }
            //   printf("\n");
            // }
            // printf("\n");
        }
    }

    //--------------------------------
    //         Make Vector x
    //--------------------------------
    for (i=0;i<ROW;i++){
        for (j=0;j<COL;j++){
            x[ROW*i+j] = u[i][j];
        }
    }
    //--------------------------------
    //        Make Vector b
    //--------------------------------
    for (i=0;i<ROW;i++){
        for (j=0;j<COL;j++){
          if (i==0 || i==ROW-1 || j==0 || j==COL-1)
            b[ROW*i+j] = 0;//1/(2*pow(pi,2))*func(i,j,dx,dy);
          else
              b[ROW*i+j] = dx*dx*func(i,j,dx,dy);

          // printf(" i:%d, j:%d, b[i][j] : %f\n",i,j,b[ROW*i+j] );
        }
    }
}

//------------------------------------------------------------
//              Matrix Calcuation Functions
//------------------------------------------------------------
double norm_L2(double *a)//計算L2向量範數
{
    int i;
    double sum = 0;

    for (i=0;i<ROW*COL;i++){
        sum = sum + pow(a[i],2);
    }
    return sqrt(sum);
}

void vmdot(double **A,double *x,double *b)//矩陣-向量乘法b=A*x
{
    int i,j;

    for (i=0;i<ROW*COL;i++){
            b[i] = 0;
    }

    for (i=0;i<ROW*COL;i++){
        for (j=0;j<ROW*COL;j++){
            b[i] = b[i] + A[i][j]*x[j];
        }

    }
}

double vvdot(double *a, double *b)//兩項量內積
{
    int i;
    double c = 0;

    for (i=0;i<ROW*COL;i++){
        c = c + a[i]*b[i];
    }

    return c;
}
