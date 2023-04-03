

#if defined(_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0;}
inline omp_int_t omp_get_num_threads() { return 1;}
#endif

#define NUM_THREADS 1
#define SIZE 128
#define NORM_CALC_INTERVAL 500
#define MAX_ITERS 5000
#define RESIDUAL_FACTOR 10000

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <cstring>
#include <cmath>
#include "utils.h"

double ResidualNorm(long N, double* u, double* f, double h){
    double norm_sum = 0.0;
    for (long i = 1; i < N+1; i++){
        for(long j = 1; j < N+1; j++){
            long point = i*(N+2) + j;
            long c1 = (i-1)*(N+2) + j;
            long c2 = (i)*(N+2) + j-1;

            long c3 = (i+1)*(N+2) + j;
            long c4 = (i)*(N+2) + j+1;
            double temp_sum = 0.0;
            temp_sum -= (u[c1] + u[c2] + u[c3] + u[c4]);
            temp_sum += 4*u[point];
            temp_sum /= (h*h);
            norm_sum += (temp_sum - f[point])*(temp_sum - f[point]);
        }
    }
    return std::sqrt(norm_sum);
}


double* Jacobi_2D_OMP(long N, double* u, double* f, double h, long max_iters, long res_factor){
    printf("\nRunning Jacobi_2D\n");

    double* u_next = (double*) malloc((N+2) * (N+2) * sizeof(double)); // (N+2)*(N+2)
    // one time intialization to all zeros
    for(long new_i = 0; new_i < N+2; new_i++){
        for(long new_j = 0; new_j < N+2; new_j++){
            u_next[new_i*(N+2) + new_j] = 0.0;
        }
    }
    long iter = 0;
    double initial_residual_norm = -1;
    double residual_norm = -1;
    Timer t;
    t.tic();
    while(iter < max_iters){

        #pragma omp parallel for num_threads(NUM_THREADS)
        for(long i = 1; i < N+1; i++){
            for(long j = 1; j < N+1; j++){
                double temp_sum = 0.0;
                temp_sum += h*h*f[i*(N+2) + j];
                temp_sum += u[(i-1)*(N+2) + (j)];
                temp_sum += u[(i)*(N+2) + (j-1)];
                temp_sum += u[(i+1)*(N+2) + (j)];
                temp_sum += u[(i)*(N+2) + (j+1)];
                temp_sum /= 4.0;
                u_next[i*(N+2) + j] = temp_sum;
            }
        }

        double* temp = u;
        u = u_next;
        u_next = temp;
        iter += 1;
        if(iter==1){
            residual_norm = ResidualNorm(/*N=*/N, /*u=*/u, /*f=*/f, /*h=*/h);
            initial_residual_norm = residual_norm;
        }

        if (iter%NORM_CALC_INTERVAL == 0){
            residual_norm = ResidualNorm(/*N=*/N, /*u=*/u, /*f=*/f, /*h=*/h);
            printf("J: Iter: %d Residual norm: %f\n", iter, residual_norm);
            if(initial_residual_norm/residual_norm > res_factor){
                printf("J: Early exit at iteration: %d due to sufficient reduction in residual norm.\n", iter);
                break;
            };
        }
    }
    double time = t.toc();
    printf("\nJ: Initial residual norm: %f\n", initial_residual_norm);
    printf("J: Final residual norm: %f\n", residual_norm);
    printf("J: Total iterations: %d, Time taken: %f sec, Time per iteration: %f sec\n", iter, time, time/iter);
    free(u_next);
    return u;
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

    int N = SIZE;
    // Note that for the linear system: Au = f, we utilize the padding of zeros
    // to avoid conditional statements inside for loops. Thus we are trading some
    // extra space to avoid overheads of conditional statements.
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

    u = Jacobi_2D_OMP(/*N=*/N, /*u=*/u, /*f=*/f, /*h=*/h, /*max_iters=*/MAX_ITERS, /*res_factor=*/RESIDUAL_FACTOR);

    free(u);
    free(f);

    return 0;
}
