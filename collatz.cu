//This is the main section of the program, where all of the kernels and functions used in the program are defined.
//I'll try to explain everything that I've done in the comments.

#include <iostream>
#include <string>
#include <algorithm>
#include <stdint.h>
#include <math.h>
#include <chrono>
#include <thread>
#include "collatz_old_algs.h"

/*
 * This is the kernel for running a Mesh-Collatz sequence with the "repeated k steps" algorithm, and it's what will be actively used.
 * Arguments:
 * repeatDepth <int> - the depth of the sieve; how many steps are repeated at once
 * sharedMemLevel <int> - how much shared memory to use (calculated in the CPU function)
 * sieveSize <long> - the size of the sieve, equal to 2^repeatDepth
 * c <array of longs> - a blank array to store all of the delay information from each thread into
 * s <array of longs> - stores the outcome after repeatDepth steps of numbers modulo the sieveSize.
 * t <array of bytes> - stores the number of 3x+1 steps of numbers modulo the sieveSize
 * cycleLeads <array of longs> - stores a SORTED list of all the numbers that lead to a cycle within repeatDepth steps
 * cycleLeadBounds <array of longs> - stores some info about the cycleLeads array: {smallest value, largest value, size of the array}
 * cycleLeadDelays <array of bytes> - stores how many steps it takes for each number in cycleLeads to reach a cycle
 * minN <long> - the lowest value of |n| to test
 * maxN <long> - the highest value of |n| to test
 */
__global__
void meshColGPURepeatedSteps(int repeatDepth, int sharedMemLevel, long long sieveSize, long long *c, long long *s, char *t, long long *cycleLeads, long long *cycleLeadBounds, char *cycleLeadDelays, long long minN, long long maxN) {
    // d_repeated stores the delay for the current thread
    long long d_repeated = 0;
    // pregenerating the powers of 3 under 2^63 for faster calculations
    __shared__ long long pows3[40];
    if (threadIdx.x == 0) {
        pows3[0] = 1;
        for (int i = 1; i < 40; i++) {
            pows3[i] = pows3[i-1] * 3;
        }
    }
    // Here's all the cycleLeadBounds stuff
    long long cycleLeadMax = cycleLeadBounds[1];
    long long cycleLeadMin = cycleLeadBounds[0];
    int cycleLeadSize = cycleLeadBounds[2]-1;

    //The only changes between each sharedMemLevel are that less things use sharedMemory, so I'll only comment on the first one    
    if (sharedMemLevel == 2) { //Shared Cycle Leads *and* Sieve
        //Using a size of 512 for cycleLeads
        const int sharedSize = 512;
        __shared__ long long sharedCycleLeads[sharedSize];
        __shared__ char sharedCycleLeadDelays[sharedSize];
        for (int i = threadIdx.x; i < cycleLeadSize; i += blockDim.x) {
            sharedCycleLeads[i] = cycleLeads[i];
            sharedCycleLeadDelays[i] = cycleLeadDelays[i];
        }
        //...and a size of 2048 for the sieve
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
            loopStart2: long long x = sign * i; //The label is so that we can run the loop with i and -i.
            while (true) {
                if (x >= cycleLeadMin && x <= cycleLeadMax) { // If x is within the range of cycleLeads we do a binary search
                    //This is self explanatory
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
                d_repeated+=repeatDepth+sharedT[(x % sieveSize + sieveSize) % sieveSize]; //We want to double-count 3x+1 steps since it's really 2 steps in one
                if (x >= sieveSize || x < 0) {
                    //todo: account for 64-bit overflow, e.g. n = 8528817511 goes over 64-bits, but is lucky enough to not do that in one go
                    /*if ((INT64_MAX - s[x%sieveSize])/powf(3,t[x%sieveSize]) < x/sieveSize) {
                        printf("Warning: Overflow\nn = %llu, x = %llu, calculating %llu*%llu+%llu\n", i, x, (long long)powf(3,t[x%sieveSize]), x/sieveSize, s[x%sieveSize]);
                    }*/

                    /*
                     * The algorithm is simpler than it looks. For a sieveSize of 2^n, with our previous x as 2^n*a + b we want x = 3^T(b)*a + S(b), as defined by the algorithm.
                     * To find a we just do x mod sieveSize, and for b we just do floor(x / sieveSize). However, we need some fixes to prevent mods from being negative and division from truncating towards zero:
                     * We replace a mod b (a % b) with (a % b + b) % b
                     * And we replace a / b with (a - (b - 1)) / b for negatives only
                     */
                    x = pows3[sharedT[(x % sieveSize + sieveSize) % sieveSize]]*((x>0 ? x : (x - (sieveSize - 1))) / sieveSize)+sharedS[(x % sieveSize + sieveSize) % sieveSize];
                } else { //If x is small things are a lot easier
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

/*
 * This is the function for running the Mesh-Collatz sequence with the "repeated k steps" algorithm. It's currently hardcoded at m = 0.
 * Arguments:
 * repeatDepth <int> - how many steps to repeat in each iteration; the sieve size is 2^repeatDepth
 * N <long> - the number to test up to the absolute value of
 * blocks <int> - the number of CUDA blocks to use
 * threads <int> - the number of CUDA threads to use
 */
float meshColRepeatedSteps(int repeatDepth, long long N, int blocks, int threads) {
    //Hardcoded m = 0 for now

    //Initialize c and CUDA Events
    long long *c;
    cudaMallocManaged(&c, sizeof(long long)*blocks*threads);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    //Find the numbers that go to 1 within repeatDepth steps.
    long long *cycleLeads;
    cudaMallocManaged(&cycleLeads, sizeof(long long)*100000000);
    char *cycleLeadDelays;
    cudaMallocManaged(&cycleLeadDelays, sizeof(char)*100000000);
    long long *cycleStarts;
    cudaMallocManaged(&cycleStarts, sizeof(long long)*100000000);
    //Hardcoding cycleLeads and cycleStarts for now.
    cycleLeads[0] = -17L;
    cycleLeads[1] = -5L;
    cycleLeads[2] = -1L;
    cycleLeads[3] = 0L;
    cycleLeads[4] = 1L;
    cycleStarts[0] = -17L;
    cycleStarts[1] = -5L;
    cycleStarts[2] = -1L;
    cycleStarts[3] = 0L;
    cycleStarts[4] = 1L;
    long long cycleLeadNext = 5;
    long long cycleLeadNow = 5;
    long long cycleLeadPrev = 0;
    //cycleLeadPrev is where the last iteration started looking at, cycleLeadNow is where this one started looking at, and cycleLeadNext is where we're looking at now
    long long *cycleLeadBounds;
    cudaMallocManaged(&cycleLeadBounds, sizeof(long long)*3);
    //printf("Generating cycle leads\n");
    for (int iters = 0; iters < repeatDepth; iters++) {
        for (int i = cycleLeadPrev; i < cycleLeadNow; i++) {
            //This is pretty inefficient since we're blatantly checking if any numbers are double counted but that also doesn't matter
            if (std::find(cycleLeads, cycleLeads + cycleLeadNext, (cycleLeads[i] << 1))==cycleLeads + cycleLeadNext) {
                //This part checks for the reverse of x/2 (no +m yet)
                cycleLeads[cycleLeadNext] = cycleLeads[i] << 1; cycleLeadNext++;
            }
            if (((cycleLeads[i] % 3 + 3) % 3 == 2) && (std::find(cycleLeads, cycleLeads + cycleLeadNext, (2 * cycleLeads[i]-1)/3)==cycleLeads + cycleLeadNext)) {
                //This part checks for the reverse of (3x+1)/2 (no +m yet)
                cycleLeads[cycleLeadNext] = (2 * cycleLeads[i] - 1) / 3; cycleLeadNext++;
            }
        }
        cycleLeadPrev = cycleLeadNow;
        cycleLeadNow = cycleLeadNext;
    }
    std::sort(cycleLeads, cycleLeads + cycleLeadNow); //Sort the array
    cycleLeadBounds[0] = cycleLeads[0];
    cycleLeadBounds[1] = cycleLeads[cycleLeadNow - 1];
    cycleLeadBounds[2] = cycleLeadNow;

    //This goes through everything in the array and manually calculates the delay
    for (int i = 0; i < cycleLeadNow; i++) {
        long long x = cycleLeads[i];
        cycleLeadDelays[i] = 0;
        while (true) {
            //A lot of this is also hardcoded for now, but it's basically a CPU version of the regular algorithm
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


    //Now we generate the sieve.
    //S saves the trajectory, T saves the number of 3x+1 steps. See the kernel for the theory, but we can use these to skip repeatDepth steps at a time.
    long long *s;
    char *t; //We're using char because we know t <= repeatDepth and repeatDepth <= 255
    cudaMallocManaged(&s, sizeof(long long)<<repeatDepth);
    cudaMallocManaged(&t, sizeof(char)<<repeatDepth);
    //This could be done on GPU, but if we're sticking with a 2^16 sieve it might not be worth it
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

    //In preparation for the GPU, we initialize c[i];
    for (int i = 0; i < blocks*threads; i++) {
        c[i] = 0;
    }

    //Here's the logic for all of the shared memory things.
    int sharedMemLevel = 0;
    if (2048 >= cycleLeadNow) {
        sharedMemLevel = 1;
    }
    if (512 >= cycleLeadNow && repeatDepth <= 11) {
        sharedMemLevel = 2;
    }

    //Enter hyperspeed
    cudaEventRecord(start);
    meshColGPURepeatedSteps<<<blocks, threads>>>(repeatDepth, sharedMemLevel, 1<<repeatDepth, c, s, t, cycleLeads, cycleLeadBounds, cycleLeadDelays, 1, N);
    cudaEventRecord(stop);


    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    
    long long delayTot = c[0];
    for (int i = 1; i < blocks*threads; i++) {
        delayTot += c[i]; //Add up the c array to find the total delay
    }
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Elapsed time: " << std::to_string(milliseconds) << "ms" << std::endl;
    std::cout << "Tot delay: " << std::to_string(delayTot) << std::endl;
    
    //Make sure to free all the memory at the end
    cudaFree(c);
    cudaFree(s);
    cudaFree(t);
    cudaFree(cycleLeads);
    cudaFree(cycleStarts);
    cudaFree(cycleLeadBounds);
    cudaFree(cycleLeadDelays);

    return milliseconds;
}

/*
 * This is the kernel for running a Mesh-Collatz sequence with the traditional algorithm.
 * Arguments:
 * c <array of longs> - a blank array to add the delays of each thread to after program execution
 * minN <long> - the smallest value of |n| to test
 * maxN <long> - the largest value of |n| to test
 * meshOffset <int> - the Mesh offset to test
 * cycleStarts <array of longs> - a SORTED list of numbers that are the low points of a cycle
 * cycleStartSize <int> - the number of elements in cycleStarts (sizeof is weird)
 */
__global__
void meshColGPUShortcut(long long *c, long long minN, long long maxN, int meshOffset, long long *cycleStarts, int cycleStartSize) {
    //Saving the smallest and largest values for a binary search
    long long cycleStartMax = cycleStarts[cycleStartSize-1];
    long long cycleStartMin = cycleStarts[0];
    __shared__ long long sharedCycleStarts[256]; //Shared arrays add more speed - as this is just for testing purposes, we don't care if there's more than 256 cycles
    for (int i = threadIdx.x; i <= cycleStartSize; i+=blockDim.x) {
        sharedCycleStarts[i] = cycleStarts[i]; //initializing values
    }
    __syncthreads();
    for (long long i = minN+threadIdx.x+blockIdx.x*blockDim.x; i <= maxN; i+=blockDim.x*gridDim.x) {
        int sign = 1;
        loopStart: long long x = sign * i; //Using a label here will cause the loop to go through x and -x.
        if (x == 0) printf(""); //For some reason not adding this would cause the program to crash.
        while (true) { //The only way to exit the loop is through a goto
            if (cycleStartMin <= x && x <= cycleStartMax) { //x is within range -> binary search
                //This section is self-explanatory
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
            ++c[threadIdx.x+blockIdx.x*blockDim.x]; //Add 1 to the delay value
            if (x % 2) {
                ++c[threadIdx.x+blockIdx.x*blockDim.x]; //We're using the 3x+1 definition, so add 1 again for the shortcut
                x *= 3; x++; x >>= 1; x += meshOffset; //(3x + 1)/2 + m. Right shift is faster than division
            } else {
                x >>= 1; x += meshOffset;//x/2 + m. Right shift is faster than division
            }
        }
        endOfLoopShortcut:;
        if (sign * i > 0) { //This will go back to the beginning when sign = 1 and i is nonzero
            sign = -1;
            goto loopStart;
        }
    }
}

/*
 * This is the function for running a Mesh-Collatz sequence with the traditional algorithm. Currently hardcoded for m = 0.
 * Arguments:
 * N <long> - the number to test up to (by absolute value)
 * blocks <int> - the number of CUDA blocks to use
 * threads <int> - the number of CUDA threads to use in each block
 */
float meshColShortcut(long long N, int blocks, int threads) {
    std::cout << "Testing up to n = +/- " << std::to_string(N) << std::endl; //Print testing range - again, this is just for testing purposes
    cudaEvent_t start, stop; //Add CUDA events for benchmarking
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    long long *c; //Initializing our c array
    cudaMallocManaged(&c, sizeof(long long)*blocks*threads);
    for (int i = 0; i < blocks*threads; i++) {
        c[i] = 0;
    }

    //As this is for testing, m = 0 is hardcoded in this cycleStarts array
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
    cudaEventSynchronize(stop); //Wait for everything to finish

    long long delayTot = c[0];
    for (int i = 1; i < blocks*threads; i++) {
        delayTot += c[i]; //Add up the c array to find the delay total
    }
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Elapsed time: " << std::to_string(milliseconds) << "ms" << std::endl;
    std::cout << "Tot delay: " << std::to_string(delayTot) << std::endl; //Sometimes there's a mismatch because of 64-bit overflow here but not in the more complex algorithm
    
    
    cudaFree(c);
    cudaFree(cycleStarts); //Free memory
    return milliseconds;
}
int main() {
    //Right now this just runs a benchmark of all numbers up to +/- 10B
    std::cout << "Mesh-Collatz Searcher v0.1.2\nRunning Benchmark..." << std::endl;
    long long N = 10000000000LL;
    printf("No Sieve - Mesh\n");
    meshColShortcut(N, 512, 512);
    printf("2^16 Sieve - Mesh\n"); //For now a 2^16 sieve seems the fastest. This will have to be benchmarked later for every Mesh offset
    meshColRepeatedSteps(16, N, 512, 512);
}