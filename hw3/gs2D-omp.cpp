

#if defined(_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0;}
inline omp_int_t omp_get_num_threads() { return 1;}
#endif

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <cstring>
#include <cmath>
#include "utils.h"

double ResidualNorm(long N, double* A, double* u, double* f){
    double norm_sum = 0.0;
    for(long i = 0; i < (N+2)*(N+2); i++){
        double temp_sum = 0.0;
        for(long j = 0; j < (N+2)*(N+2); j++){
            temp_sum += A[i*(N+2)*(N+2) + j]*u[j];
        }
        norm_sum += (temp_sum - f[i])*(temp_sum - f[i]);
    }
    return std::sqrt(norm_sum);
}


double* GS_2D_OMP(long N, double* A, double* u, double* f, double h, long max_iters, long res_factor){
    printf("\nRunning GS_2D based on Red-Black coloring using OpenMP\n");
    long iter = 0;
    double initial_residual_norm = -1;
    double residual_norm = -1;
    Timer t;
    t.tic();
    while(iter < max_iters){

        #pragma omp parallel for num_threads(16)
        for(long i = 1; i < N+1; i++){
            // update the red points based on prev black points
            for(long j = 1 + (i-1)%2; j < N+1; j+=2){
                // printf("\n Red Point i: %ld j: %ld \n", i, j);
                double temp_sum = 0.0;
                temp_sum += h*h*f[i*(N+2) + j];
                temp_sum += u[(i-1)*(N+2) + (j)];
                temp_sum += u[(i)*(N+2) + (j-1)];
                temp_sum += u[(i+1)*(N+2) + (j)];
                temp_sum += u[(i)*(N+2) + (j+1)];
                temp_sum /= 4.0;
                u[i*(N+2) + j] = temp_sum;
            }
        }

        #pragma omp parallel for num_threads(16)
        for(long i = 1; i < N+1; i++){
            // update the black points based on new red points
            for(long j = 1 + (i)%2; j < N+1; j+=2){
                // printf("\n Black Point i: %ld j: %ld \n", i, j);
                double temp_sum = 0.0;
                temp_sum += h*h*f[i*(N+2) + j];
                temp_sum += u[(i-1)*(N+2) + (j)];
                temp_sum += u[(i)*(N+2) + (j-1)];
                temp_sum += u[(i+1)*(N+2) + (j)];
                temp_sum += u[(i)*(N+2) + (j+1)];
                temp_sum /= 4.0;
                u[i*(N+2) + j] = temp_sum;
            }
        }

        iter += 1;
        if(iter==1){
            residual_norm = ResidualNorm(N, A, u, f);
            initial_residual_norm = residual_norm;
        }

        if (iter%100 == 0){
            residual_norm = ResidualNorm(N, A, u, f);
            printf("GS: Iter: %d Residual norm: %f\n", iter, residual_norm);
            if(initial_residual_norm/residual_norm > res_factor){
                printf("GS: Early exit at iteration: %d due to sufficient reduction in residual norm.\n", iter);
                break;
            };
        }
    }
    double time = t.toc();
    printf("\nGS: Initial residual norm: %f\n", initial_residual_norm);
    printf("GS: Final residual norm: %f\n", residual_norm);
    printf("GS: Total iterations: %d, Time taken: %f sec, Time per iteration: %f sec\n", iter, time, time/iter);

    return u;
}


double* GS_2D_Vanilla_Seq(long N, double* A, double* u, double* f, double h, long max_iters, long res_factor){
    printf("\nRunning GS2D in Sequential Mode\n");
    long iter = 0;
    double initial_residual_norm = -1;
    double residual_norm = -1;
    Timer t;
    t.tic();
    while(iter < max_iters){
        
        // perform a row-based scan so that the ordering
        // is suitable for gauss-seidel. Especially, when
        // updating u_{i,j}, the previous iterations would
        // have already handled 
        for(long i = 1; i < N+1; i++){
            for(long j = 1; j < N+1; j++){
                double temp_sum = 0.0;
                temp_sum += h*h*f[i*(N+2) + j];
                temp_sum += u[(i-1)*(N+2) + (j)];
                temp_sum += u[(i)*(N+2) + (j-1)];
                temp_sum += u[(i+1)*(N+2) + (j)];
                temp_sum += u[(i)*(N+2) + (j+1)];
                temp_sum /= 4.0;
                u[i*(N+2) + j] = temp_sum;
            }
        }

        iter += 1;

        if(iter==1){
            residual_norm = ResidualNorm(N, A, u, f);
            initial_residual_norm = residual_norm;
        }

        if (iter%100 == 0){
            residual_norm = ResidualNorm(N, A, u, f);
            printf("GS: Iter: %d Residual norm: %f\n", iter, residual_norm);
            if(initial_residual_norm/residual_norm > res_factor){
                printf("GS: Early exit at iteration: %d due to sufficient reduction in residual norm.\n", iter);
                break;
            };
        }

    }
    double time = t.toc();
    printf("\nGS: Initial residual norm: %f\n", initial_residual_norm);
    printf("GS: Final residual norm: %f\n", residual_norm);
    printf("GS: Total iterations: %d, Time taken: %f sec, Time per iteration: %f sec\n", iter, time, time/iter);

    return u;
}

void PrintA(long N, double* A){
    printf("\nPrinting A (without padding)\n");

    for (long ri = 1; ri < N+1; ri++){
        for(long rj = 1; rj < N+1; rj++){
            long row = ri*(N+2) + rj;
            printf("\nROW %d\n", row);
            for (long ci = 1; ci < N+1; ci++){
                for(long cj = 1; cj < N+1; cj++){
                    long col = ci*(N+2) + cj;
                    printf("%f ", A[row*(N+2)*(N+2) + col]);
                }
            }
        }
    }
}

void Printu(long N, double* u){
    printf("\nPrinting u (without padding)\n");
    for (long i = 1; i < N+1; i++) {
        for (long j = 1; j < N+1; j++) {
            printf("%f ", u[i*(N+2) + j]);
        }
    }
    printf("\n");
}

int main(int argc, char** argv) {

    int N = 100;
    // Note that for the linear system: Au = f, we utilize the padding of zeros
    // to avoid conditional statements inside for loops. Thus we are trading some
    // extra space to avoid overheads of conditional statements.
    double* A = (double*) malloc((N+2) * (N+2) * (N+2) * (N+2) * sizeof(double)); // (N+2) * (N+2) * (N+2) * (N+2)
    double* u = (double*) malloc((N+2) * (N+2) * sizeof(double)); // (N+2) x (N+2)
    double* f = (double*) malloc((N+2) * (N+2) * sizeof(double)); // (N+2) x (N+2)

    // Initialize data.
    // Note: We consider a row-major order for vectors in this problem

    double h = 1.0/(N+1);

    printf("Initializing u\n");

    for (long i = 0; i < N+2; i++){
        for(long j = 0; j < N+2; j++){
            u[i*(N+2) + j] = 0.0;
        }
    }

    printf("Initializing f\n");
    // Since we are primarily interested in the non-boundary values, we use
    // boundary padding for residual computation.
    for (long i = 0; i < N+2; i++){
        for(long j = 0; j < N+2; j++){
            if(i==0 or j==0 or i==N+1 or j==N+1) {
                f[i*(N+2) + j] = 0.0;
            }
            else f[i*(N+2) + j] = 1.0;
        }
    }

    printf("Initializing A\n");
    for (long i = 0; i < (N+2)*(N+2); i++){
        for(long j = 0; j < (N+2)*(N+2); j++){
            A[i*(N+2)*(N+2) + j] = 0.0;
        }
    }
    printf("Filling A\n");
    // Since we are primarily interested in the non-boundary values, we use
    // boundary padding for residual computation.
    for (long i = 1; i < N+1; i++){
        for(long j = 1; j < N+1; j++){
            long row = i*(N+2) + j;
            long c1 = (i-1)*(N+2) + j;
            long c2 = (i)*(N+2) + j-1;
            long c3 = i*(N+2) + j;
            long c4 = (i+1)*(N+2) + j;
            long c5 = (i)*(N+2) + j+1;
            // printf("row: %ld c1: %ld c2: %ld c3: %ld c4: %ld\n", row, c1, c2, c3, c4);
            A[row*(N+2)*(N+2) + c1] = -1.0/(h*h);
            A[row*(N+2)*(N+2) + c2] = -1.0/(h*h);
            A[row*(N+2)*(N+2) + c3] = +4.0/(h*h);
            A[row*(N+2)*(N+2) + c4] = -1.0/(h*h);
            A[row*(N+2)*(N+2) + c5] = -1.0/(h*h);
        }
    }

    u = GS_2D_OMP(/*N=*/N, /*A=*/A, /*u=*/u, /*f=*/f, /*h=*/h, /*max_iters=*/5000, /*res_factor=*/10000);

    free(A);
    free(u);
    free(f);

    return 0;
}
