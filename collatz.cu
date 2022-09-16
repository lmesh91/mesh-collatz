#include <iostream>
#include <string>
__global__
void collatzGPU(long long *c, long long minN, long  long maxN) {
    for (long long i = minN+threadIdx.x+blockIdx.x*blockDim.x; i < maxN; i+=blockDim.x*gridDim.x) {
        long long x = i;
        ++c[0];
        while (x != 1) {
            if (x % 2) {
                x *= 3;
                x++;
            } else {
                x /= 2;
            }
        }
    }
}
int main() {
    std::cout << "Mesh-Collatz Searcher v0.0.1" << std::endl;
    long long N = 1000000000000LL;
    std::cout << "Testing up to n = " << std::to_string(N) << std::endl;
    long long *c;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMallocManaged(&c, sizeof(long long));
    c[0]=1;

    cudaEventRecord(start);
    
    collatzGPU<<<256, 256>>>(c, 1, N);
    cudaEventRecord(stop);

    cudaFree(c);

    cudaDeviceSynchronize();

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Elapsed time: " << std::to_string(milliseconds) << "ms" << std::endl;
}