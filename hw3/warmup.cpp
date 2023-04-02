

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

void f(int i){
    // ideally the factor should be 1000 to represent time in milli-seconds,
    //  however one can set it to a smaller number (for ex: 50) for faster simulations.
    sleep(i/1000);
}


int main(int argc, char** argv) {
    printf("maximum number of threads = %d\n", omp_get_num_threads());
    int N = 100;
    int i = 0;
    std::cout << "Loop 1" << std::endl;
    #pragma omp parallel for schedule(static) num_threads(2)
    for (i = 1; i < N; i++) {
        printf("Thread %d is doing iteration %d.\n", omp_get_thread_num(), i);
        f(i);
    }

    std::cout << "Loop 2" << std::endl;
    #pragma omp parallel for schedule(static) num_threads(2)
    for (i = 1; i < N; i++) {
        printf("Thread %d is doing iteration %d.\n", omp_get_thread_num(), i);
        f(N-i);
    }

    return 0;
}

// 
// `nowait` Implementation
// 
// int main(int argc, char** argv) {
//     printf("maximum number of threads = %d\n", omp_get_num_threads());
//     int N = 100;
//     int i = 0;
//     std::cout << "Loop 1" << std::endl;
//     #pragma omp parallel num_threads(2)
//     {
//         #pragma omp for nowait
//         for (i = 1; i < N; i++) {
//             printf("Thread %d is doing iteration %d in loop 1.\n", omp_get_thread_num(), i);
//             f(i);
//         }

//         std::cout << "Loop 2" << std::endl;
    
//         #pragma omp for nowait
//         for (i = 1; i < N; i++) {
//             printf("Thread %d is doing iteration %d in loop 2.\n", omp_get_thread_num(), i);
//             f(N-i);
//         }
//     }

//     return 0;
// }
