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

