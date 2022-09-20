#include <iostream>
#include <string>
#include <algorithm>
#include <stdint.h>
#include <math.h>
#include <chrono>
#include <thread>
#include "collatz_old_algs.h"
__global__
void meshColGPURepeatedSteps(int repeatDepth, int sharedMemLevel, long long sieveSize, long long *c, long long *s, char *t, long long *cycleLeads, long long *cycleLeadBounds, char *cycleLeadDelays, long long minN, long long maxN) {
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
    long long cycleLeadMin = cycleLeadBounds[0];
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
            int sign = 1;
            loopStart2: long long x = sign * i;
            while (true) {
                if (x >= cycleLeadMin && x <= cycleLeadMax) {
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
                d_repeated+=repeatDepth+sharedT[(x % sieveSize + sieveSize) % sieveSize];
                //todo: account for 64-bit overflow, e.g. n = 8528817511 goes over 64-bits, but is lucky enough to not do that in one go
                if (x >= sieveSize || x < 0) {
                    /*if ((INT64_MAX - s[x%sieveSize])/powf(3,t[x%sieveSize]) < x/sieveSize) {
                        printf("Warning: Overflow\nn = %llu, x = %llu, calculating %llu*%llu+%llu\n", i, x, (long long)powf(3,t[x%sieveSize]), x/sieveSize, s[x%sieveSize]);
                    }*/
                    // a mod b always rounding down: (a % b + b) % b
                    // a/b always rounding down: (a - (b - 1)) / b for negatives
                    x = pows3[sharedT[(x % sieveSize + sieveSize) % sieveSize]]*((x>0 ? x : (x - (sieveSize - 1))) / sieveSize)+sharedS[(x % sieveSize + sieveSize) % sieveSize];
                } else {
                    x = sharedS[x];
                }
            }
            endOfLoop2:;
            if (sign * i > 0) { //Cleverly, this also won't get stuck when i = 0
                sign = -1;
                goto loopStart2;
            }
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
            int sign = 1;
            loopStart1: long long x = sign * i;
            while (true) {
                if (x >= cycleLeadMin && x <= cycleLeadMax) { 
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
                d_repeated+=repeatDepth+t[(x % sieveSize + sieveSize) % sieveSize];
                //todo: account for 64-bit overflow, e.g. n = 8528817511 goes over 64-bits, but is lucky enough to not do that in one go
                if (x >= sieveSize || x < 0) {
                    /*if ((INT64_MAX - s[x%sieveSize])/powf(3,t[x%sieveSize]) < x/sieveSize) {
                        printf("Warning: Overflow\nn = %llu, x = %llu, calculating %llu*%llu+%llu\n", i, x, (long long)powf(3,t[x%sieveSize]), x/sieveSize, s[x%sieveSize]);
                    }*/
                    // a mod b always rounding down: (a % b + b) % b
                    // a/b always rounding down: (a - (b - 1)) / b for negatives
                    x = pows3[t[(x % sieveSize + sieveSize) % sieveSize]]*((x>0 ? x : (x - (sieveSize - 1))) / sieveSize)+s[(x % sieveSize + sieveSize) % sieveSize];
                } else {
                    x = s[x];
                }
            }
            endOfLoop1:;
            if (sign * i > 0) { //Cleverly, this also won't get stuck when i = 0
                sign = -1;
                goto loopStart1;
            }
        }
    }
    if (sharedMemLevel == 0) { //Unified Cycle Leads and Sieve
        for (long long i = minN+threadIdx.x+blockIdx.x*blockDim.x; i <= maxN; i+=blockDim.x*gridDim.x) {
            int sign = 1;
            loopStart0: long long x = sign * i;
            while (true) {
                if (x >= cycleLeadMin && x <= cycleLeadMax) { 
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
                d_repeated+=repeatDepth+t[(x % sieveSize + sieveSize) % sieveSize];
                //todo: account for 64-bit overflow, e.g. n = 8528817511 goes over 64-bits, but is lucky enough to not do that in one go
                if (x >= sieveSize || x < 0) {
                    /*if ((INT64_MAX - s[x%sieveSize])/powf(3,t[x%sieveSize]) < x/sieveSize) {
                        printf("Warning: Overflow\nn = %llu, x = %llu, calculating %llu*%llu+%llu\n", i, x, (long long)powf(3,t[x%sieveSize]), x/sieveSize, s[x%sieveSize]);
                    }*/
                    // a mod b always rounding down: (a % b + b) % b
                    // a/b always rounding down: (a - (b - 1)) / b for negatives
                    x = pows3[t[(x % sieveSize + sieveSize) % sieveSize]]*((x>0 ? x : (x - (sieveSize - 1))) / sieveSize)+s[(x % sieveSize + sieveSize) % sieveSize];
                } else {
                    x = s[x];
                }
            }
            endOfLoop0:;
            if (sign * i > 0) { //Cleverly, this also won't get stuck when i = 0
                sign = -1;
                goto loopStart0;
            }
        }
    }
    c[threadIdx.x+blockIdx.x*blockDim.x]=d_repeated;
}

float meshColRepeatedSteps(int repeatDepth, long long N, int blocks, int threads) {
    //Hardcoded m = 0 for now
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
    cycleLeads[0] = -17L;
    cycleLeads[1] = -5L;
    cycleLeads[2] = -1L;
    cycleLeads[3] = 0L;
    cycleLeads[4] = 1L;
    long long *cycleStarts;
    cudaMallocManaged(&cycleStarts, sizeof(long long)*100000000);
    cycleStarts[0] = -17L;
    cycleStarts[1] = -5L;
    cycleStarts[2] = -1L;
    cycleStarts[3] = 0L;
    cycleStarts[4] = 1L;
    long long cycleLeadNext = 5;
    long long cycleLeadNow = 5;
    long long cycleLeadPrev = 0;
    long long *cycleLeadBounds;
    cudaMallocManaged(&cycleLeadBounds, sizeof(long long)*3);
    cycleLeadBounds[0] = 1;
    cycleLeadBounds[1] = 1;
    //printf("Generating cycle leads\n");
    for (int iters = 0; iters < repeatDepth; iters++) {
        for (int i = cycleLeadPrev; i < cycleLeadNow; i++) {
            if (std::find(cycleLeads, cycleLeads + cycleLeadNext, (cycleLeads[i] << 1))==cycleLeads + cycleLeadNext) {
                cycleLeads[cycleLeadNext] = cycleLeads[i] << 1; cycleLeadNext++;
            }
            if (((cycleLeads[i] % 3 + 3) % 3 == 2) && (std::find(cycleLeads, cycleLeads + cycleLeadNext, (2 * cycleLeads[i]-1)/3)==cycleLeads + cycleLeadNext)) {
                cycleLeads[cycleLeadNext] = (2 * cycleLeads[i] - 1) / 3; cycleLeadNext++;
            }
        }
        cycleLeadPrev = cycleLeadNow;
        cycleLeadNow = cycleLeadNext;
    }
    cycleLeadBounds[2] = cycleLeadNow;

    std::sort(cycleLeads, cycleLeads + cycleLeadNow);
    cycleLeadBounds[0] = cycleLeads[0];
    cycleLeadBounds[1] = cycleLeads[cycleLeadNow - 1];
    cudaMallocManaged(&cycleLeadDelays, sizeof(char)*100000000);
    cycleLeadDelays[0] = 1;
    for (int i = 0; i < cycleLeadNow; i++) {
        long long x = cycleLeads[i];
        cycleLeadDelays[i] = 0;
        while (true) {
            if (-17 <= x && x <= 1) {
                int low = 0;
                int high = 5;  
                while (low != high) {
                    int mid = (low+high)/2;
                    if (x == cycleStarts[mid]) {
                        goto endOfLoopShortcutInCPU;
                    } else if (x > cycleStarts[mid]) {
                        low = mid + 1;
                    } else {
                        high = mid;
                    }
                }
            }
            ++cycleLeadDelays[i];
            if (x % 2) {
                ++cycleLeadDelays[i];
                x *= 3; x++; x >>= 1; //x += meshOffset;//(3x+1)/2
            } else {
                x >>= 1; //x += meshOffset;//Still dividing by 2; right shift is 2 faster
            }
        }
        endOfLoopShortcutInCPU:;
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
    if (2048 >= cycleLeadNow) {
        sharedMemLevel = 1;
    }
    if (512 >= cycleLeadNow && repeatDepth <= 11) {
        sharedMemLevel = 2;
    }

    cudaEventRecord(start);
    meshColGPURepeatedSteps<<<blocks, threads>>>(repeatDepth, sharedMemLevel, 1<<repeatDepth, c, s, t, cycleLeads, cycleLeadBounds, cycleLeadDelays, 1, N);
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
    cudaFree(cycleStarts);
    cudaFree(cycleLeadBounds);
    cudaFree(cycleLeadDelays);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Elapsed time: " << std::to_string(milliseconds) << "ms" << std::endl;
    std::cout << "Tot delay: " << std::to_string(delayTot) << std::endl;

    return milliseconds;
}

__global__
void meshColGPUShortcut(long long *c, long long minN, long long maxN, int meshOffset, long long *cycleStarts, int cycleStartSize) {
    long long cycleStartMax = cycleStarts[cycleStartSize-1];
    long long cycleStartMin = cycleStarts[0];
    __shared__ long long sharedCycleStarts[256];
    for (int i = threadIdx.x; i <= cycleStartSize; i+=blockDim.x) {
        sharedCycleStarts[i] = cycleStarts[i];
    }
    __syncthreads();
    for (long long i = minN+threadIdx.x+blockIdx.x*blockDim.x; i <= maxN; i+=blockDim.x*gridDim.x) {
        int sign = 1;
        loopStart: long long x = sign * i;
        if (x == 0) printf("");
        while (true) {
            if (cycleStartMin <= x && x <= cycleStartMax) {
                int low = 0;
                int high = cycleStartSize;  
                while (low != high) {
                    int mid = (low+high)/2;
                    if (x == sharedCycleStarts[mid]) {
                        goto endOfLoopShortcut;
                    } else if (x > sharedCycleStarts[mid]) {
                        low = mid + 1;
                    } else {
                        high = mid;
                    }
                }
            }
            ++c[threadIdx.x+blockIdx.x*blockDim.x];
            if (x % 2) {
                ++c[threadIdx.x+blockIdx.x*blockDim.x];
                x *= 3; x++; x >>= 1; x += meshOffset;//(3x+1)/2
            } else {
                x >>= 1; x += meshOffset;//Still dividing by 2; right shift is 2 faster
            }
        }
        endOfLoopShortcut:;
        if (sign * i > 0) { //Cleverly, this also won't get stuck when i = 0
            sign = -1;
            goto loopStart;
        }
    }
}
float meshColShortcut(long long N, int blocks, int threads) {
    std::cout << "Testing up to n = +/- " << std::to_string(N) << std::endl;
    long long *c;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMallocManaged(&c, sizeof(long long)*blocks*threads);
    for (int i = 0; i < blocks*threads; i++) {
        c[i] = 0;
    }

    //Hardcoded m = 0 for now
    long long *cycleStarts;
    cudaMallocManaged(&cycleStarts, sizeof(long long)*4);
    cycleStarts[0] = -17L;
    cycleStarts[1] = -5L;
    cycleStarts[2] = -1L;
    cycleStarts[3] = 0L;
    cycleStarts[4] = 1L;

    cudaEventRecord(start);
    
    meshColGPUShortcut<<<blocks, threads>>>(c, 1, N, 0, cycleStarts, 5);
    cudaEventRecord(stop);


    cudaDeviceSynchronize();

    cudaEventSynchronize(stop);
    
    long long delayTot = c[0];
    for (int i = 1; i < blocks*threads; i++) {
        delayTot += c[i];
    }
    cudaFree(c);
    cudaFree(cycleStarts);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Elapsed time: " << std::to_string(milliseconds) << "ms" << std::endl;
    std::cout << "Tot delay: " << std::to_string(delayTot) << std::endl;

    return milliseconds;
}
int main() {
    std::cout << "Mesh-Collatz Searcher v0.1.1\nRunning Benchmark..." << std::endl;
    long long N = 10000000000LL;
    printf("Warmup Round\n");
    meshColShortcut(N, 512, 512);
    meshColRepeatedSteps(16, N, 512, 512);
    printf("No Sieve - Mesh\n");
    meshColShortcut(N, 512, 512);
    printf("2^16 Sieve - Mesh\n");
    meshColRepeatedSteps(16, N, 512, 512);
}