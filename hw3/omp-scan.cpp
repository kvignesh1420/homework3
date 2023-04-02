#if defined(_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0;}
inline omp_int_t omp_get_num_threads() { return 1;}
#endif

#include <algorithm>
#include <stdio.h>
#include <math.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

// void scan_for(long* prefix_sum, const long* A, long n) {
//   int p = omp_get_num_threads();
//   // int p = 64;
//   int t = omp_get_thread_num();
//   // Fill out parallel scan: One way to do this is array into p chunks
//   // Do a scan in parallel on each chunk, then share/compute the offset
//   // through a shared vector and update each chunk by adding the offset
//   // in parallel
//   if (n == 0) return;
//   prefix_sum[0] = 0;
//   long chunk_size = n/(long)p;
//   printf("Num of chunk: %d, Chunk size: %ld\n", p, chunk_size);
//   for(int chunk = 0; chunk < p; chunk++){
//     for(int idx = chunk_size*(chunk); idx < chunk_size*(chunk+1); idx++){
//       if(idx == 0) continue;
//       // if(idx == chunk*chunk_size) prefix_sum[idx] = A[idx-1];
//       else prefix_sum[idx] = prefix_sum[idx-1] + A[idx-1];
//     }
//   }
// }

void scan_omp(long* prefix_sum, const long* A, long n) {
  // int p = omp_get_num_threads();
  // if openmp fails to get the correct number of threads, you can try
  // hardcoding the value as follows.
  int p = 16;

  // Fill out parallel scan: One way to do this is array into p chunks
  // Do a scan in parallel on each chunk, then share/compute the offset
  // through a shared vector and update each chunk by adding the offset
  // in parallel
  if (n == 0) return;
  prefix_sum[0] = 0;
  long chunk_size = n/(long)p;
  // printf("Num of chunk: %d, Chunk size: %ld\n", p, chunk_size);

  #pragma omp parallel num_threads(p)
  {  
    #pragma omp for
    for(int chunk = 0; chunk < p; chunk++){
      for(int idx = chunk_size*(chunk); idx < chunk_size*(chunk+1); idx++){
        if(idx == 0) continue;
        if(idx == chunk*chunk_size) prefix_sum[idx] = A[idx-1];
        else prefix_sum[idx] = prefix_sum[idx-1] + A[idx-1];
      }
    }
  }

  long* offsets = (long*) malloc(p * sizeof(long));
  offsets[0] = 0;
  for(int i=1; i < p; i++){
    offsets[i] = offsets[i-1] + prefix_sum[chunk_size*i-1];
  }

  #pragma omp parallel num_threads(p)
  {  
    #pragma omp for
    for(int chunk = 0; chunk < p; chunk++){
      for(int idx = chunk_size*(chunk); idx < chunk_size*(chunk+1); idx++){
        prefix_sum[idx] += offsets[chunk];
      }
    }
  }

  free(offsets);

}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();
  for (long i = 0; i < N; i++) B1[i] = 0;

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
