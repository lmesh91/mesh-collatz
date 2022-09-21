//This section of the codebase contains some older algorithms that weren't adapted for Mesh offsets.

#include <iostream>
#include <string>
#include <algorithm>
#include <stdint.h>
#include <math.h>
#include <chrono>
#include <thread>
#include "collatz_old_algs.h"
__global__
void collatzGPURepeatedSteps(int repeatDepth, int sharedMemLevel, long long sieveSize, long long *c, long long *s, char *t, long long *cycleLeads, long long *cycleLeadBounds, char *cycleLeadDelays, long long minN, long long maxN) {
    long long d_repeated = 0;
    //__shared__ long long sharedCycLeads[];
    //long long cycleLeadMin = cycleLeadBounds[0];
    __shared__ long long pows3[40];
    if (threadIdx.x == 0) {
        pows3[0] = 1;
        for (int i = 1; i < 40; i++) {
            pows3[i] = pows3[i-1] * 3;
        }
    }
    long long cycleLeadMax = cycleLeadBounds[1];
    int cycleLeadSize = cycleLeadBounds[2]-1;
    if (sharedMemLevel == 2) { //Shared Cycle Leads *and* Sieve
        const int sharedSize = 512;
        __shared__ long long sharedCycleLeads[sharedSize];
        __shared__ char sharedCycleLeadDelays[sharedSize];
        for (int i = threadIdx.x; i < cycleLeadSize; i += blockDim.x) {
            sharedCycleLeads[i] = cycleLeads[i];
            sharedCycleLeadDelays[i] = cycleLeadDelays[i];
        }
        const int sharedSieveSize = 2048;
        __shared__ long long sharedS[sharedSieveSize];
        __shared__ char sharedT[sharedSieveSize];
        for (int i = threadIdx.x; i < sieveSize; i++) {
            sharedS[i] = s[i];
            sharedT[i] = t[i];
        }
        __syncthreads();
        for (long long i = minN+threadIdx.x+blockIdx.x*blockDim.x; i <= maxN; i+=blockDim.x*gridDim.x) {
            long long x = i;
            while (x != 1) {
                if (x <= cycleLeadMax) { //add x > cycleLeadBounds[0] when doing mesh-collatz
                    int low = 0;
                    int high = cycleLeadSize;  
                    while (low != high) {
                        int mid = (low+high)/2;
                        if (x == sharedCycleLeads[mid]) {
                            d_repeated += sharedCycleLeadDelays[mid];
                            goto endOfLoop2;
                        } else if (x > sharedCycleLeads[mid]) {
                            low = mid + 1;
                        } else {
                            high = mid;
                        }
                    }
                }
                d_repeated+=repeatDepth+t[x%sieveSize];
                //todo: account for 64-bit overflow, e.g. n = 8528817511 goes over 64-bits, but is lucky enough to not do that in one go
                if (x >= sieveSize || x < 0) {
                    /*if ((INT64_MAX - s[x%sieveSize])/powf(3,t[x%sieveSize]) < x/sieveSize) {
                        printf("Warning: Overflow\nn = %llu, x = %llu, calculating %llu*%llu+%llu\n", i, x, (long long)powf(3,t[x%sieveSize]), x/sieveSize, s[x%sieveSize]);
                    }*/
                    x = pows3[sharedT[x%sieveSize]]*(x/sieveSize)+sharedS[x%sieveSize];
                } else {
                    x = sharedS[x];
                }
            }
            endOfLoop2:;
        }
    }
    if (sharedMemLevel == 1) { //Shared Cycle Leads, Unified Sieve
        const int sharedSize = 2048;
        __shared__ long long sharedCycleLeads[sharedSize];
        __shared__ char sharedCycleLeadDelays[sharedSize];
        for (int i = threadIdx.x; i < cycleLeadSize; i += blockDim.x) {
            sharedCycleLeads[i] = cycleLeads[i];   
            sharedCycleLeadDelays[i] = cycleLeadDelays[i];
        }
        __syncthreads();
        for (long long i = minN+threadIdx.x+blockIdx.x*blockDim.x; i <= maxN; i+=blockDim.x*gridDim.x) {
            long long x = i;
            while (x != 1) {
                if (x <= cycleLeadMax) { //add x > cycleLeadBounds[0] when doing mesh-collatz
                    int low = 0;
                    int high = cycleLeadSize;  
                    while (low != high) {
                        int mid = (low+high)/2;
                        if (x == sharedCycleLeads[mid]) {
                            d_repeated += sharedCycleLeadDelays[mid];
                            goto endOfLoop1;
                        } else if (x > sharedCycleLeads[mid]) {
                            low = mid + 1;
                        } else {
                            high = mid;
                        }
                    }
                }
                d_repeated+=repeatDepth+t[x%sieveSize];
                //todo: account for 64-bit overflow, e.g. n = 8528817511 goes over 64-bits, but is lucky enough to not do that in one go
                if (x >= sieveSize || x < 0) {
                    /*if ((INT64_MAX - s[x%sieveSize])/powf(3,t[x%sieveSize]) < x/sieveSize) {
                        printf("Warning: Overflow\nn = %llu, x = %llu, calculating %llu*%llu+%llu\n", i, x, (long long)powf(3,t[x%sieveSize]), x/sieveSize, s[x%sieveSize]);
                    }*/
                    x = pows3[t[x%sieveSize]]*(x/sieveSize)+s[x%sieveSize];
                } else {
                    x = s[x];
                }
            }
            endOfLoop1:;
        }
    }
    if (sharedMemLevel == 0) { //Unified Cycle Leads and Sieve
        for (long long i = minN+threadIdx.x+blockIdx.x*blockDim.x; i <= maxN; i+=blockDim.x*gridDim.x) {
            long long x = i;
            while (x != 1) {
                if (x <= cycleLeadMax) { //add x > cycleLeadBounds[0] when doing mesh-collatz
                    int low = 0;
                    int high = cycleLeadSize;  
                    while (low != high) {
                        int mid = (low+high)/2;
                        if (x == cycleLeads[mid]) {
                            d_repeated += cycleLeadDelays[mid];
                            goto endOfLoop0;
                        } else if (x > cycleLeads[mid]) {
                            low = mid + 1;
                        } else {
                            high = mid;
                        }
                    }
                }
                d_repeated+=repeatDepth+t[x%sieveSize];
                //todo: account for 64-bit overflow, e.g. n = 8528817511 goes over 64-bits, but is lucky enough to not do that in one go
                if (x >= sieveSize || x < 0) {
                    /*if ((INT64_MAX - s[x%sieveSize])/powf(3,t[x%sieveSize]) < x/sieveSize) {
                        printf("Warning: Overflow\nn = %llu, x = %llu, calculating %llu*%llu+%llu\n", i, x, (long long)powf(3,t[x%sieveSize]), x/sieveSize, s[x%sieveSize]);
                    }*/
                    x = pows3[t[x%sieveSize]]*(x/sieveSize)+s[x%sieveSize];
                } else {
                    x = s[x];
                }
            }
            endOfLoop0:;
        }
    }
    c[threadIdx.x+blockIdx.x*blockDim.x]=d_repeated;
}

float collatzRepeatedSteps(int repeatDepth, long long N, int blocks, int threads) {
    //std::cout << "Testing up to n = " << std::to_string(N) << std::endl;
    long long *c;
    cudaMallocManaged(&c, sizeof(long long)*blocks*threads);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    //Find the numbers that go to 1 within repeatDepth steps.
    long long *cycleLeads;
    cudaMallocManaged(&cycleLeads, sizeof(long long)*100000000);
    char *cycleLeadDelays;
    cycleLeads[0] = 1;
    long long cycleLeadNext = 1;
    long long cycleLeadNow = 1;
    long long cycleLeadPrev = 0;
    long long *cycleLeadBounds;
    cudaMallocManaged(&cycleLeadBounds, sizeof(long long)*3);
    cycleLeadBounds[0] = 1;
    cycleLeadBounds[1] = 1;
    //printf("Generating cycle leads\n");
    for (int iters = 0; iters < repeatDepth; iters++) {
        for (int i = cycleLeadPrev; i < cycleLeadNow; i++) {
            cycleLeads[cycleLeadNext] = cycleLeads[i] << 1; cycleLeadNext++;
            if (cycleLeadBounds[0] < cycleLeads[cycleLeadNext-1]) {
                cycleLeadBounds[0] = cycleLeads[cycleLeadNext-1];
            }
            if (cycleLeadBounds[1] < cycleLeads[cycleLeadNext-1]) {
                cycleLeadBounds[1] = cycleLeads[cycleLeadNext-1];
            }
            if ((cycleLeads[i] % 3 == 2) && (std::find(cycleLeads, cycleLeads + cycleLeadNext, (2 * cycleLeads[i]-1)/3)==cycleLeads + cycleLeadNext)) {
                cycleLeads[cycleLeadNext] = (2 * cycleLeads[i] - 1) / 3; cycleLeadNext++;
                if (cycleLeadBounds[0] < cycleLeads[cycleLeadNext-1]) {
                    cycleLeadBounds[0] = cycleLeads[cycleLeadNext-1];
                }
                if (cycleLeadBounds[1] < cycleLeads[cycleLeadNext-1]) {
                    cycleLeadBounds[1] = cycleLeads[cycleLeadNext-1];
                }
            }
        }
        cycleLeadPrev = cycleLeadNow;
        cycleLeadNow = cycleLeadNext;
    }
    cycleLeadBounds[2] = cycleLeadNow;

    std::sort(cycleLeads, cycleLeads + cycleLeadNow);
    cudaMallocManaged(&cycleLeadDelays, sizeof(char)*100000000);
    cycleLeadDelays[0] = 1;
    for (int i = 0; i < cycleLeadNow; i++) {
        long long x = cycleLeads[i];
        cycleLeadDelays[i] = 0;
        while (x != 1) {
            ++cycleLeadDelays[i];
            if (x % 2) {
                ++cycleLeadDelays[i];
                x *= 3; x++; x >>= 1; //(3x+1)/2
            } else {
                x >>= 1; //Still dividing by 2; right shift is 2 faster
            }
        }
    }


    //printf("Generating Sieve\n");
    //Now we generate the sieve.
    //S saves the trajectory, T saves the number of 3x+1 steps. We can show every number is equal to 3^T*(x/2^r)+S
    long long *s;
    char *t;
    cudaMallocManaged(&s, sizeof(long long)<<repeatDepth);
    cudaMallocManaged(&t, sizeof(char)<<repeatDepth);
    //Todo: Do this on GPU
    for (long long i = 0; i < 1<<repeatDepth; i++) {
        long long x = i;
        for (int j = 0; j < repeatDepth; j++) {
            if (x % 2) {
                x *= 3; ++x; x >>= 1; ++t[i];
            } else {
                x >>= 1;
            }
        }
        s[i] = x;
    }

    //It's GPU time!
    for (int i = 0; i < blocks*threads; i++) {
        c[i] = 0;
    }
    //printf("Starting\n");

    int sharedMemLevel = 0;
    if (2048 <= cycleLeadNow) {
        sharedMemLevel = 1;
    }
    if (512 <= cycleLeadNow && repeatDepth <= 11) {
        sharedMemLevel = 2;
    }

    cudaEventRecord(start);
    collatzGPURepeatedSteps<<<blocks, threads>>>(repeatDepth, sharedMemLevel, 1<<repeatDepth, c, s, t, cycleLeads, cycleLeadBounds, cycleLeadDelays, 1, N);
    cudaEventRecord(stop);


    cudaDeviceSynchronize();

    cudaEventSynchronize(stop);
    
    long long delayTot = c[0];
    for (int i = 1; i < blocks*threads; i++) {
        delayTot += c[i];
    }
    cudaFree(c);
    cudaFree(s);
    cudaFree(t);
    cudaFree(cycleLeads);
    cudaFree(cycleLeadBounds);
    cudaFree(cycleLeadDelays);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Elapsed time: " << std::to_string(milliseconds) << "ms" << std::endl;
    std::cout << "Tot delay: " << std::to_string(delayTot) << std::endl;

    return milliseconds;
}

__global__
void collatzGPUShortcut(long long *c, long long minN, long  long maxN) {
    for (long long i = minN+threadIdx.x+blockIdx.x*blockDim.x; i <= maxN; i+=blockDim.x*gridDim.x) {
        long long x = i;
        //++c[0];
        while (x != 1) {
            ++c[threadIdx.x+blockIdx.x*blockDim.x];
            if (x % 2) {
                ++c[threadIdx.x+blockIdx.x*blockDim.x];
                x *= 3; x++; x >>= 1; //(3x+1)/2
            } else {
                x >>= 1; //Still dividing by 2; right shift is 2 faster
            }
        }
    }
}
float collatzShortcut(long long N, int blocks, int threads) {
    //std::cout << "Testing up to n = " << std::to_string(N) << std::endl;
    long long *c;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMallocManaged(&c, sizeof(long long)*blocks*threads);
    for (int i = 0; i < blocks*threads; i++) {
        c[i] = 0;
    }

    cudaEventRecord(start);
    
    collatzGPUShortcut<<<blocks, threads>>>(c, 1, N);
    cudaEventRecord(stop);


    cudaDeviceSynchronize();

    cudaEventSynchronize(stop);
    
    long long delayTot = c[0];
    for (int i = 1; i < blocks*threads; i++) {
        delayTot += c[i];
    }
    cudaFree(c);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Elapsed time: " << std::to_string(milliseconds) << "ms" << std::endl;
    std::cout << "Tot delay: " << std::to_string(delayTot) << std::endl;

    return milliseconds;
}