#include <iostream>
#include <string>
__global__
void collatzGPURepeatedSteps(int repeatDepth, long long sieveSize, long long *c, long long *s, long long *t, long long *cycleLeads, long long *cycleLeadBounds, long long *pows3, long long minN, long long maxN) {
    for (long long i = minN+threadIdx.x+blockIdx.x*blockDim.x; i <= maxN; i+=blockDim.x*gridDim.x) {
        long long x = i;
        while (x != 1) {
            if (x < cycleLeadBounds[1]) { //add x > cycleLeadBounds[0] when doing mesh-collatz
                for (int j = 0; j < cycleLeadBounds[2]; j++) { //todo: sort list and binary search
                    if (cycleLeads[j] == x) {
                        while (x != 1) {
                            ++c[threadIdx.x+blockIdx.x*blockDim.x];
                            if (x % 2) {
                                ++c[threadIdx.x+blockIdx.x*blockDim.x];
                                x *= 3; x++; x >>= 1; //(3x+1)/2
                            } else {
                                x >>= 1; //Still dividing by 2; right shift is 2 faster
                            }
                        }
                        goto endOfLoop;
                    }
                }
            }
            c[threadIdx.x+blockIdx.x*blockDim.x]+=repeatDepth+t[x%sieveSize];
            if (x >= sieveSize || x < 0) {
                x = pows3[t[x%sieveSize]]*(x/sieveSize)+s[x%sieveSize];
            } else {
                x = s[x];
            }
            endOfLoop: continue;
        }
    }
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
    cudaMallocManaged(&cycleLeads, sizeof(long long)*1000000000);
    cycleLeads[0] = 1;
    long long cycleLeadNext = 1;
    long long cycleLeadNow = 1;
    long long cycleLeadPrev = 0;
    long long *cycleLeadBounds;
    cudaMallocManaged(&cycleLeadBounds, sizeof(long long)*3);
    cycleLeadBounds[0] = 1;
    cycleLeadBounds[1] = 1;
    printf("Generating cycle leads\n");
    for (int iters = 0; iters <= repeatDepth; iters++) {
        for (int i = cycleLeadPrev; i < cycleLeadNow; i++) {
            cycleLeads[cycleLeadNext] = cycleLeads[i] << 1; cycleLeadNext++;
            if (cycleLeadBounds[0] < cycleLeads[cycleLeadNext-1]) {
                cycleLeadBounds[0] = cycleLeads[cycleLeadNext-1];
            }
            if (cycleLeadBounds[1] < cycleLeads[cycleLeadNext-1]) {
                cycleLeadBounds[1] = cycleLeads[cycleLeadNext-1];
            }
            if (cycleLeads[i] % 3 == 2) {
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

    printf("Generating SIEVE\n");
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

    //Precalculating powers of 3 to save time later.
    long long *pows3;
    cudaMallocManaged(&pows3, sizeof(long long)*40);
    pows3[0] = 1;
    for (int i = 1; i < 39; i++) {
        pows3[i] = pows3[i-1] * 3;
    }

    //It's GPU time!
    for (int i = 0; i < blocks*threads; i++) {
        c[i] = 0;
    }
    printf("Starting");
    cudaEventRecord(start);
    collatzGPURepeatedSteps<<<blocks, threads>>>(repeatDepth, 1<<repeatDepth, c, s, t, cycleLeads, cycleLeadBounds, pows3, 1, N);
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
void collatzGPUNaive(long long *c, long long minN, long  long maxN) {
    for (long long i = minN+threadIdx.x+blockIdx.x*blockDim.x; i <= maxN; i+=blockDim.x*gridDim.x) {
        long long x = i;
        //++c[0];
        while (x != 1) {
            ++c[threadIdx.x+blockIdx.x*blockDim.x];
            if (x % 2) {
                x *= 3; x++;
            } else {
                x >>= 1; //Still dividing by 2; right shift is 2 faster
            }
        }
    }
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

float collatzNaive(long long N, int blocks, int threads) {
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
    
    collatzGPUNaive<<<blocks, threads>>>(c, 1, N);
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
    long long N = 1000000000LL;
    //collatzRepeatedSteps(15, N, 256, 512); //switch to 256x512 when stable
    collatzShortcut(N, 256, 512);
}