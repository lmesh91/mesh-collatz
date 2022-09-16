#include <iostream>
#include <string>
__global__
void collatzGPUNaive(long long *c, long long minN, long  long maxN) {
    for (long long i = minN+threadIdx.x+blockIdx.x*blockDim.x; i < maxN; i+=blockDim.x*gridDim.x) {
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
    for (long long i = minN+threadIdx.x+blockIdx.x*blockDim.x; i < maxN; i+=blockDim.x*gridDim.x) {
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

/*
Benchmarks (up to 1B):
All with 256x512

collatzGPUNaive, Windows - 440.4 ms
collatzGPUShortcut, Windows - 352.2 ms
*/

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
    if (delayTot != 203234783274) {
        std::cout << "Error" << std::endl;
        return -1;
    } else {
        std::cout << "Elapsed time: " << std::to_string(milliseconds) << "ms" << std::endl;
    }
    //std::cout << "Avg delay: " << std::to_string((double)delayTot/(double)N) << std::endl;

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
    if (delayTot != 203234783274) {
        std::cout << "Error" << std::endl;
        return -1;
    } else {
        std::cout << "Elapsed time: " << std::to_string(milliseconds) << "ms" << std::endl;
    }
    //std::cout << "Avg delay: " << std::to_string((double)delayTot/(double)N) << std::endl;

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
    std::cout << "Mesh-Collatz Searcher v0.0.2" << std::endl;
    long long N = 1000000000LL;
    bench(N, 256, 512, 50);
}