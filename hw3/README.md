## Homework 3

### Setup
The `linserv1.cims.nyu.edu` host was used for running all the experiments. The details of the processor are as follows:

```
[vk2115@linserv1 homework3]$ lscpu
Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                64
On-line CPU(s) list:   0-63
Thread(s) per core:    2
Core(s) per socket:    8
Socket(s):             4
NUMA node(s):          8
Vendor ID:             AuthenticAMD
CPU family:            21
Model:                 1
Model name:            AMD Opteron(TM) Processor 6272
Stepping:              2
CPU MHz:               2099.939
BogoMIPS:              4199.87
Virtualization:        AMD-V
L1d cache:             16K
L1i cache:             64K
L2 cache:              2048K
L3 cache:              6144K
NUMA node0 CPU(s):     0-7
NUMA node1 CPU(s):     8-15
NUMA node2 CPU(s):     32-39
NUMA node3 CPU(s):     40-47
NUMA node4 CPU(s):     48-55
NUMA node5 CPU(s):     56-63
NUMA node6 CPU(s):     16-23
NUMA node7 CPU(s):     24-31
Flags:                 fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc art rep_good nopl nonstop_tsc extd_apicid amd_dcm aperfmperf pni pclmulqdq monitor ssse3 cx16 sse4_1 sse4_2 popcnt aes xsave avx lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs xop skinit wdt fma4 nodeid_msr topoext perfctr_core perfctr_nb cpb hw_pstate ssbd rsb_ctxsw ibpb vmmcall retpoline_amd arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold
```

### Question 1: Open-MP Warmup

Consider the following code and assume it is executes by two threads. The
for-loops are executed in two chunks, one per thread, and the independent functions f(i) take i
milliseconds.

```c++
#pragma omp parallel
{
    // loop 1
    #pragma omp for schedule(static)
    for (i = 1; i < n; i++)
        f(i)
    
    // loop 2
    #pragma omp for schedule(static)
    for (i = 1; i < n; i++)
        f(n-i)
}
```

1. How long would each threads spend to execute the parallel region? How much of that time
would be spent in waiting for the other thread?

- As the schedule is static, OpenMP seems to divide the workload (`n-1` iterations per loop) into chunks of (almost) equal size among the two threads:
$
\begin{align}
c_1 = (n-1)//2 \\
c_2 = (n-1) - c_1
\end{align}
$

The execution time for each thread per loop (based on the statically allocated chunk) would be:

$
\begin{align}
t_1 = 1 + \cdots + c_1 = \frac{c_1*(c_1+1)}{2} \\
t_2 = c_1+1 \cdots + n-1 = c_1*c_2 + \frac{c_2*(c_2+1)}{2} \\
\end{align}
$
Now, since `omp for` is a fork-join model of thread execution, both the threads work on their chunk of work for loop 1 in parallel. However, they need to complete their chunk of work for loop 1 before they are allocated the workload for loop 2. To this end, the faster thread has to wait for $t_2 - t_1$ milliseconds before being allocated the chunks for loop 2.

I have written a sample program to validate this observation.
```bash
$ g++ -fopenmp warmup.cpp && ./a.out
```

2. How would the execution time of each thread change if we used schedule(static,1) for
both loops?

- With this schedule, the threads will be allocated chunks of size 1 (iteration) for execution. Once the respective chunk is finished, a new chunk is allotted. To this end, we can observe an alternative chunk exection pattern. For instance: 

```
Loop 2
Thread 0 is doing iteration 1 in loop 2.
Thread 1 is doing iteration 2 in loop 2.
Thread 0 is doing iteration 3 in loop 2.
Thread 1 is doing iteration 4 in loop 2.
Thread 0 is doing iteration 5 in loop 2.
Thread 1 is doing iteration 6 in loop 2.
Thread 0 is doing iteration 7 in loop 2.
Thread 1 is doing iteration 8 in loop 2.
```

Thus, the execution times are modified to (Assuming $n-1$ is odd):
$
\begin{align}
t_1 = 1 + 3 + \cdots + n-1 \\
t_2 = 2 + 4 + \cdots + n-2 \\
\end{align}
$
To this end, the faster thread has to wait for $t_2 - t_1$ milliseconds before being allocated the chunks for loop 2, which is relatively smaller than that of the previous case.

3. Would it improve if we used schedule(dynamic,1) instead?

- There is not much improvement with `schedule(dynamic,1)` as the chunk size is still 1 and the chunk allocation pattern converges to an alternative one as before.

4. Is there an OpenMP directive that allows to eliminate the waiting time and how much would
the threads take when using this clause?

- Yes, we can use the `nowait` directive to avoid the waiting times. With this approach, there is zero to minimal waiting times for the threads as the faster thread in loop 1 will be the slower one in loop 2. Thus, they both finish at approximately the same time.

### Question 2: Parallel Scan

| N | Sequential Latency(sec) | Parallel: 8 threads Latency (sec) |
|---|-------------------------|-----------------------------------|
| $10^8$ | $1.156152$ | $0.250289$ |
| $2 \times 10^8$ | $2.720552$ | $0.828265$ |
| $4 \times 10^8$ | $6.027957$ | $1.540014$ |

| N | Sequential Latency(sec) | Parallel: 16 threads Latency (sec) |
|---|-------------------------|-----------------------------------|
| $10^8$ | $1.156152$ | $0.245381$ |
| $2 \times 10^8$ | $2.720552$ | $0.789866$ |
| $4 \times 10^8$ | $6.027957$ | $0.929338$ |


| N | Sequential Latency(sec) | Parallel: 64 threads Latency (sec) |
|---|-------------------------|-----------------------------------|
| $10^8$ | $1.156152$ | $0.225652$ |
| $2 \times 10^8$ | $2.720552$ | $0.592096$ |
| $4 \times 10^8$ | $6.027957$ | $1.234108$ |

The increase in number of threads from 16 to 64 with $N=4 \times 10^8$ highlights the issues with memory contention and access patterns. However, for smaller $N$ values, the speedup with increasing the number of threads is evident.

### Question 3: Jacobi and Gauss-Seidel 2D

#### Jacobi 2D

Max iterations = $5000$, Reduction factor threshold = $10000$, Norm calculation interval = $500$.

The norm calculation interval indicates the number of iterations after which the residual norm is calculated to check for convergence. It is set to a higher value of $500$ as the convergence is slow and to prevent frequent switches from parallel execution to norm calculation.

| N   | Thread count| Iters | Total Time (sec)| Time per Iter (sec)  |
|----:|------------:|------:|----------------:|---------------------:|
| 128 |       1     | 5000  |   0.253694      |   0.000051           |
| 128 |       8     | 5000  |   0.078674      |   0.000016           |
| 128 |      16     | 5000  |   0.101497      |   0.000020           |
| 128 |      64     | 5000  |   1.609469      |   0.000558           |
| 256 |      1      | 5000  |   0.942978      |   0.000189           |
| 256 |      8      | 5000  |   0.215450      |   0.000043           |
| 256 |     16      | 5000  |   0.184045      |   0.000037           |
| 256 |     64      | 5000  |   2.204146      |   0.000441           |
| 512 |      1      | 5000  |   13.345154     |   0.002669           |
| 512 |      8      | 5000  |   1.759324      |   0.000352           |
| 512 |     16      | 5000  |   1.018521      |   0.000204           |
| 512 |     64      | 5000  |   2.350867      |   0.000470           |
|1024 |      1      | 5000  |   63.725291     |   0.012745           |
|1024 |      8      | 5000  |   8.861325      |   0.001772           |
|1024 |     16      | 5000  |   4.573597      |   0.000915           |
|1024 |     64      | 5000  |   4.532769      |   0.000907           |

#### GS 2D

Max iterations = $5000$, Reduction factor threshold = $10000$, Norm calculation interval = $500$.

| N   | Thread count| Iters | Total Time (sec)| Time per Iter (sec)  |
|----:|------------:|------:|----------------:|---------------------:|
| 128 |       1     | 5000  |   0.338091      |   0.000068           |
| 128 |       8     | 5000  |   0.132346      |   0.000026           |
| 128 |      16     | 5000  |   0.214242      |   0.000043           |
| 128 |      64     | 5000  |   3.245481      |   0.000649           |
| 256 |       1     | 5000  |   1.218507      |   0.000244           |
| 256 |       8     | 5000  |   0.285945      |   0.000057           |
| 256 |      16     | 5000  |   0.321591      |   0.000064           |
| 256 |      64     | 5000  |   5.097657      |   0.001020           |
| 512 |       1     | 5000  |   9.266678      |   0.001853           |
| 512 |       8     | 5000  |   1.245816      |   0.000249           |
| 512 |      16     | 5000  |   0.904956      |   0.000181           |
| 512 |      64     | 5000  |   3.987627      |   0.000798           |
|1024 |       1     | 5000  |   40.155386     |   0.008031           |
|1024 |       8     | 5000  |   4.382397      |   0.000876           |
|1024 |      16     | 5000  |   2.494571      |   0.000499           |
|1024 |      64     | 5000  |   7.528113      |   0.001506           |


**Remarks:** 

- For both Jacobi and GS, using $8$ or $16$ threads gave the best speedup. 

- The highest speedup when using $64$ threads is obtained with $N=1024$ for Jacobi 2D. For other smaller cases, increasing the thread count to $64$ resulted in heavy contention for memory between threads. Thus, resulting in slowdowns.

