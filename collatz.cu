#include <iostream>
#include <string>
#include <algorithm>
#include <stdint.h>
#include <math.h>
__global__
void collatzGPURepeatedSteps(int repeatDepth, long long sieveSize, long long *c, long long *s, long long *t, long long *cycleLeads, long long *cycleLeadBounds, long long *cycleLeadDelays, long long SHARED_MEM_LIMIT, long long minN, long long maxN) {
    long long d_repeated = 0;
    long long cycleLeadMax = cycleLeadBounds[1];
    long long cycleLeadSize = cycleLeadBounds[2]-1;
    extern __shared__ long long sharedCycLeads[];
    for (int i = threadIdx.x; i < SHARED_MEM_LIMIT; i += blockDim.x) {
        sharedCycLeads[i] = cycleLeads[i];
    }
    long long sharedCycLeadMax = sharedCycLeads[SHARED_MEM_LIMIT-1];
    __syncthreads();
    //long long cycleLeadMin = cycleLeadBounds[0];
    for (long long i = minN+threadIdx.x+blockIdx.x*blockDim.x; i <= maxN; i+=blockDim.x*gridDim.x) {
        long long x = i;
        while (x != 1) {
            if (x <= cycleLeadMax) { //add x > cycleLeadBounds[0] when doing mesh-collatz
                if (SHARED_MEM_LIMIT - 1 == cycleLeadSize) {
                    int low = 0;
                    int high = SHARED_MEM_LIMIT - 1;  
                    while (low != high) {
                        float partPercent = 0.5;
                        int mid = partPercent*(low)+(1-partPercent)*(high-1);
                        if (x == sharedCycLeads[mid]) {
                            d_repeated += cycleLeadDelays[mid];
                            goto endOfLoop;
                        } else if (x > sharedCycLeads[mid]) {
                            low = mid + 1;
                        } else {
                            high = mid;
                        }
                    }
                } else {
                    int low = 0;
                    int high = cycleLeadSize;
                    printf("%i < %i, %llu\n", low, high, sharedCycLeadMax);  
                        goto endOfLoop;
                    while (low != high) {
                        float partPercent = 0.5;
                        int mid = partPercent*(low)+(1-partPercent)*(high-1);
                        if (x == cycleLeads[mid]) {
                            d_repeated += cycleLeadDelays[mid];
                            goto endOfLoop;
                        } else if (x > cycleLeads[mid]) {
                            low = mid + 1;
                        } else {
                            high = mid;
                        }
                    }
                }
            }
            d_repeated+=repeatDepth+t[x%sieveSize];
            //todo: account for 64-bit overflow, e.g. n = 8528817511 goes over 64-bits, but is lucky enough to not do that in one go
            if (x >= sieveSize || x < 0) {
                /*if ((INT64_MAX - s[x%sieveSize])/powf(3,t[x%sieveSize]) < x/sieveSize) {
                    printf("Warning: Overflow\nn = %llu, x = %llu, calculating %llu*%llu+%llu\n", i, x, (long long)powf(3,t[x%sieveSize]), x/sieveSize, s[x%sieveSize]);
                }*/
                long long pow3 = 1;
                for (int i = 0; i < t[x%sieveSize]; i++) {
                    pow3 *= 3;
                }
                x = pow3*(x/sieveSize)+s[x%sieveSize];
            } else {
                x = s[x];
            }
        }
        endOfLoop:;
    }
    c[threadIdx.x+blockIdx.x*blockDim.x]=d_repeated;
}

float collatzRepeatedSteps(int repeatDepth, long long N, int blocks, int threads) {
    std::cout << "Testing up to n = " << std::to_string(N) << std::endl;
    long long *c;
    cudaMallocManaged(&c, sizeof(long long)*blocks*threads);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    //Find the numbers that go to 1 within repeatDepth steps.
    long long *cycleLeads;
    cudaMallocManaged(&cycleLeads, sizeof(long long)*100000000);
    long long *cycleLeadDelays;
    cycleLeads[0] = 1;
    long long cycleLeadNext = 1;
    long long cycleLeadNow = 1;
    long long cycleLeadPrev = 0;
    long long *cycleLeadBounds;
    cudaMallocManaged(&cycleLeadBounds, sizeof(long long)*3);
    cycleLeadBounds[0] = 1;
    cycleLeadBounds[1] = 1;
    printf("Generating cycle leads\n");
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
    cudaMallocManaged(&cycleLeadDelays, sizeof(long long)*100000000);
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

    long long origMemLimit = 1024;
    if (origMemLimit > cycleLeadNow) {
        origMemLimit = cycleLeadNow;
    }
    const long long SHARED_MEM_LIMIT = origMemLimit;

    printf("Generating Sieve\n");
    //Now we generate the sieve.
    //S saves the trajectory, T saves the number of 3x+1 steps. We can show every number is equal to 3^T*(x/2^r)+S
    long long *s, *t;
    cudaMallocManaged(&s, sizeof(long long)<<repeatDepth);
    cudaMallocManaged(&t, sizeof(long long)<<repeatDepth);
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
    printf("Starting\n");
    cudaEventRecord(start);
    collatzGPURepeatedSteps<<<blocks, threads, SHARED_MEM_LIMIT * sizeof(long long)>>>(repeatDepth, 1<<repeatDepth, c, s, t, cycleLeads, cycleLeadBounds, cycleLeadDelays, SHARED_MEM_LIMIT, 1, N);
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
void bench(long long N, int blocks, int threads, int reps) {
    std::cout << std::to_string(blocks) << "x" << std::to_string(threads) << std::endl;
    float milliseconds = 0;
    for (int i = 0; i < reps; i++) {
        milliseconds += collatzShortcut(N, blocks, threads);
    }
    std::cout << "Avg time: " << std::to_string(milliseconds / reps) << "ms" << std::endl;
}



int main() {
    std::cout << "Mesh-Collatz Searcher v0.0.3" << std::endl;
    long long N = 10000000000LL;
    collatzShortcut(N, 256, 512);
    collatzRepeatedSteps(20, N, 256, 512); //switch to 256x512 when stable
}