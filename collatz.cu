//This is the main section of the program, where all of the kernels and functions used in the program are defined.
//I'll try to explain everything that I've done in the comments.

//This file has the 128-bit integer implementation, and all of the headers needed.
#include "int129.cuh"
#include "ansi.hpp"
#include <fstream>

const long long two62 = 1LL << 62;

//This is a debug function to monitor the available memory.
bool showMemory = false;
void memcheck() {
    if (showMemory) {
        //List mem (for debugging)
        size_t free, total;
        cudaMemGetInfo( &free, &total );
        printf("%s%sFree Memory: %s%lld%s bytes.%s\n", ANSI::faint, ANSI::italic, ANSI::underline, free, ANSI::resetUnderline, ANSI::reset);
    }
}

//This structure is used to return both execution time and delay when running a search.
struct CollatzResults {
    float milliseconds;
    unsigned long long delay;

    CollatzResults(float ms, unsigned long long d) {
        milliseconds = ms;
        delay = d;
    }
};

/*
 * This is the kernel for finding all cycles under a certain size in the Mesh-Collatz sequence.
 * Arguments:
 * c <array of longs> - a blank array to add the lowest values of cycles to
 * c_size <array of longs> - a blank array where the size of c will be printed to, and misc information
 * d_index <array of bytes> - a blank array where threads hold their best cycle values
 * minN <long> - the smallest value of |n| to test
 * maxN <long> - the largest value of |n| to test
 * meshOffset <int> - the Mesh offset to test
 */
__global__
void meshColGPUCycleSearch(long long *c, long long *c_size, char *d_index, long long minN, long long maxN, int meshOffset) {
    //This time, *c stores the list of cycle values.
    //c_size[1] -> Where to add the next cycle
    //c_size[2] and [3] -> min and max search values
    const int D_SIZE = 16;
    const int HIST_SIZE = 8192;
    __syncthreads();
    for (long long i = minN+threadIdx.x+blockIdx.x*blockDim.x; i <= maxN; i+=blockDim.x*gridDim.x) {
        int sign = 1;
        loopStartCycle: long long x = sign * i; //Using a label here will cause the loop to go through x and -x.
        if (x == 0) printf(""); //This is needed for the program to work
        long long startX = i; //sign doesn't matter
        long long history[HIST_SIZE];
        history[0] = x;
        int historyVal = 1;
        while (true) { //The only way to exit the loop is through a goto
            if (startX > ((x > 0) ? x : -x)) goto endOfLoopCycle; //|x| < |x_orig|
            if (historyVal == HIST_SIZE) { //Is the history array full? We've probably reached a cycle...
                for (int z = HIST_SIZE - 2; z >= 0; z--) {
                    if (history[z] == x) {
                        long long minValue = x; //Finding the minimum value of the array
                        long long minValueAbs = (minValue > 0) ? minValue : -minValue;
                        for (int j = z; j <= HIST_SIZE - 1; j++) {
                            if (minValueAbs > ((history[j] > 0) ? history[j] : -history[j])) {
                                minValue = history[j];
                                minValueAbs = (minValue > 0) ? minValue : -minValue;
                            }
                        }
                        c[D_SIZE*(threadIdx.x + blockIdx.x * blockDim.x) + d_index[threadIdx.x + blockIdx.x * blockDim.x]] = minValue; d_index[threadIdx.x + blockIdx.x * blockDim.x]++; //Add it to the thread-only D array
                        goto endOfLoopCycle;
                    }
                    if (z == 0) {
                        printf("%s%sLong cycle in m = %i, x = %lld%s\n", ANSI::brightRed, ANSI::redBack, meshOffset, startX, ANSI::reset);
                        goto endOfLoopCycle;
                    }
                }
            }
            if (x % 2) {
                x *= 3; x++;
            }
            x >>= 1; x += meshOffset; //Just a more optimized version to execute this part
            history[historyVal] = x; ++historyVal;
        }
        endOfLoopCycle:;
        if (sign * i > 0) { //This will go back to the beginning when sign = 1 and i is nonzero
            sign = -1;
            goto loopStartCycle;
        }
    }
    __syncthreads();
    //Replace C with an array containing all of the stuff in the D arrays.
    c_size[1] = blockDim.x*gridDim.x*D_SIZE;
    for (int i = d_index[threadIdx.x + blockIdx.x * blockDim.x]; i < D_SIZE; i++) {
        c[D_SIZE*(threadIdx.x + blockIdx.x * blockDim.x) + i] = -9223372036854775807LL;
    }
    __syncthreads();
    c_size[0] = c_size[1];
}
/*
 * This is the function for finding all of the cycles of a Mesh-Collatz sequence under a certain size.
 * Arguments:
 * cycles <long*> - a reference to where the cycles will be stored
 * N <long> - the number to test up to (by absolute value)
 * blocks <int> - the number of CUDA blocks to use
 * threads <int> - the number of CUDA threads to use in each block
 * meshOffset <int> - the Mesh offset to use
 * testLargeOnly <bool> - Whether to test only for large cycles; useful for finding large cycles
 */
long long* meshColCycleSearch(long long N, int blocks, int threads, int meshOffset, bool testLargeOnly = false) {
    cudaEvent_t start, stop; //Add CUDA events for benchmarking
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    long long *c; //Initializing our c array
    cudaMallocManaged(&c, sizeof(long long)*16*512*512);

    long long *c_size; //This will show the size so we can turn it into a normal array later.
    cudaMallocManaged(&c_size, sizeof(int)*4);

    char *d_index;
    cudaMallocManaged(&d_index, blocks*threads);
    for (int i = 0; i < blocks*threads; i++) {
        d_index[i] = 0;
    }

    
    cudaEventRecord(start);
    if (testLargeOnly) {
        meshColGPUCycleSearch<<<blocks, threads>>>(c, c_size, d_index, std::abs(meshOffset)*3, N, meshOffset); //temporary change for large cycles
    } else {
        meshColGPUCycleSearch<<<blocks, threads>>>(c, c_size, d_index, 0LL, N, meshOffset);
    }
    cudaEventRecord(stop);

    cudaDeviceSynchronize();
    cudaEventSynchronize(stop); //Wait for everything to finish

    //Post-Processing Part 1: Clear out null values.
    int newPos = 0;
    for (int curPos = 0; curPos < c_size[0]; curPos++) {
        if (c[curPos] == -9223372036854775807LL) {
            continue;
        } else {
            c[newPos] = c[curPos];
            newPos++;
        }
    }

    //Post-Processing Part 2: Clear out duplicates.
    std::sort(c, c + newPos);
    for (int i = 0; i < newPos; i++) {
        for (int j = 0; j < i; j++) {
            if (c[i] == c[j]) {
                for (int k = i + 1; k < newPos; k++) {
                    c[k-1] = c[k];
                }
                newPos--;
                j--;
                if (i >= newPos) break;
            }
        }
    }
    
    long long *cycles = new long long[newPos + 1];
    cycles[0] = newPos; //the first value gives the array size
    for (int i = 0; i < newPos; i++) {
        cycles[i+1] = c[i];
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    //std::cout << "Elapsed time: " << std::to_string(milliseconds) << "ms" << std::endl;
    //std::cout << "Cycles found: " << std::to_string(*cycles) << std::endl;
    
    
    cudaFree(c);
    cudaFree(d_index);
    cudaFree(c_size); //Free memory
    return cycles;
}

/*
 * This is the kernel for running a Mesh-Collatz sequence with the "repeated k steps" algorithm, and it's what will be actively used.
 * Arguments:
 * repeatDepth <int> - the depth of the sieve; how many steps are repeated at once
 * sharedMemLevel <int> - how much shared memory to use (calculated in the CPU function)
 * sieveSize <long> - the size of the sieve, equal to 2^repeatDepth
 * c <array of unsigned longs> - a blank array to store all of the delay information from each thread into
 * s <array of longs> - stores the outcome after repeatDepth steps of numbers modulo the sieveSize.
 * t <array of bytes> - stores the number of 3x+1 steps of numbers modulo the sieveSize
 * cycleLeads <array of longs> - stores a SORTED list of all the numbers that lead to a cycle within repeatDepth steps
 * cycleLeadBounds <array of longs> - stores some info about the cycleLeads array: {smallest value, largest value, size of the array}
 * cycleLeadDelays <array of bytes> - stores how many steps it takes for each number in cycleLeads to reach a cycle
 * minN <long> - the lowest value of |n| to test
 * maxN <long> - the highest value of |n| to test
 */
__global__
void meshColGPURepeatedSteps(int repeatDepth, int sharedMemLevel, long long sieveSize, unsigned long long *c, long long *s, char *t, long long *cycleLeads, long long *cycleLeadBounds, char *cycleLeadDelays, long long minN, long long maxN) {
    // d_repeated stores the delay for the current thread
    unsigned long long d_repeated = 0;
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
            loopStart2: int129 x = sign * i; //The label is so that we can run the loop with i and -i.
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
                d_repeated+=repeatDepth+sharedT[(((long long)(x.lo % two62) * (x.sign ? 1 : -1)) % sieveSize + sieveSize) % sieveSize]; //We want to double-count 3x+1 steps since it's really 2 steps in one
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
                    long long safeX = (long long)(x.lo % two62) * (x.sign ? 1 : -1); //x % 2^62 basically
                    x = pows3[sharedT[(safeX % sieveSize + sieveSize) % sieveSize]]*((x>0 ? x : (x - (sieveSize - 1))) >> repeatDepth)+sharedS[(safeX % sieveSize + sieveSize) % sieveSize];
                } else { //If x is small things are a lot easier
                    x = sharedS[x.lo];
                }
            }
            endOfLoop2:;
            if (sign * i > 0) { //Cleverly, this also won't get stuck when i = 0
                sign = -1;
                goto loopStart2;
            }
        }
    }
    if (sharedMemLevel == 1) { //Shared Cycle Leads, Unified Sieve - what this program focuses on
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
            loopStart1: int129 x = sign * i;
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
                d_repeated+=repeatDepth+t[((long long)(x.lo % two62) * (x.sign ? 1 : -1) % sieveSize + sieveSize) % sieveSize];
                //todo: account for 64-bit overflow, e.g. n = 8528817511 goes over 64-bits, but is lucky enough to not do that in one go
                if (x >= sieveSize || x < 0) {
                    /*if ((INT64_MAX - s[x%sieveSize])/powf(3,t[x%sieveSize]) < x/sieveSize) {
                        printf("Warning: Overflow\nn = %llu, x = %llu, calculating %llu*%llu+%llu\n", i, x, (long long)powf(3,t[x%sieveSize]), x/sieveSize, s[x%sieveSize]);
                    }*/
                    // a mod b always rounding down: (a % b + b) % b
                    // a/b always rounding down: (a - (b - 1)) / b for negatives
                    long long safeX = (long long)(x.lo % two62) * (x.sign ? 1 : -1); //x % 2^62 basically
                    x = pows3[t[(safeX % sieveSize + sieveSize) % sieveSize]]*((x>0 ? x : (x - (sieveSize - 1))) >> repeatDepth)+s[(safeX % sieveSize + sieveSize) % sieveSize];
                } else {
                    x = s[x.lo];
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
            loopStart0: int129 x = sign * i;
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
                d_repeated+=repeatDepth+t[(((long long)(x.lo % two62) * (x.sign ? 1 : -1)) % sieveSize + sieveSize) % sieveSize]; //We want to double-count 3x+1 steps since it's really 2 steps in one
                //todo: account for 64-bit overflow, e.g. n = 8528817511 goes over 64-bits, but is lucky enough to not do that in one go
                if (x >= sieveSize || x < 0) {
                    /*if ((INT64_MAX - s[x%sieveSize])/powf(3,t[x%sieveSize]) < x/sieveSize) {
                        printf("Warning: Overflow\nn = %llu, x = %llu, calculating %llu*%llu+%llu\n", i, x, (long long)powf(3,t[x%sieveSize]), x/sieveSize, s[x%sieveSize]);
                    }*/
                    // a mod b always rounding down: (a % b + b) % b
                    // a/b always rounding down: (a - (b - 1)) / b for negatives
                    long long safeX = (long long)(x.lo % two62) * (x.sign ? 1 : -1); //x % 2^62 basically
                    x = pows3[t[(safeX % sieveSize + sieveSize) % sieveSize]]*((x>0 ? x : (x - (sieveSize - 1))) >> repeatDepth)+s[(safeX % sieveSize + sieveSize) % sieveSize];
               } else {
                    x = s[x.lo];
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
 * This is the function for running the Mesh-Collatz sequence with the "repeated k steps" algorithm.
 * Arguments:
 * repeatDepth <int> - how many steps to repeat in each iteration; the sieve size is 2^repeatDepth
 * minN <long> - the number to start testing with (by absolute value)
 * maxN <long> - the number to test up to the absolute value of
 * meshOffset <int> - the Mesh offset
 * blocks <int> - the number of CUDA blocks to use
 * threads <int> - the number of CUDA threads to use
 */
CollatzResults* meshColRepeatedSteps(int repeatDepth, long long minN, long long maxN, int meshOffset, int blocks, int threads) {
    //Initialize c and CUDA Events
    unsigned long long *c;
    cudaMallocManaged(&c, sizeof(unsigned long long)*blocks*threads);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    //Find the numbers that go to 1 within repeatDepth steps.
    const int MAX_MALLOC = 16384;
    long long *cycleLeads;
    cudaMallocManaged(&cycleLeads, sizeof(long long)*MAX_MALLOC);
    char *cycleLeadDelays;
    cudaMallocManaged(&cycleLeadDelays, MAX_MALLOC);
    long long *cycleStarts;
    cudaMallocManaged(&cycleStarts, sizeof(long long)*MAX_MALLOC);
    cycleStarts = meshColCycleSearch(100000+100000*(meshOffset > 0 ? meshOffset : -meshOffset), blocks, threads, meshOffset);
    int numCycles = cycleStarts[0];
    for (int i = 0; i < numCycles; i++) {
        cycleStarts[i] = cycleStarts[i+1];
        cycleLeads[i] = cycleStarts[i];
    }
    long long cycleStartMin = cycleStarts[0];
    long long cycleStartMax = cycleStarts[numCycles-1];
    long long cycleLeadNext = numCycles;
    long long cycleLeadNow = numCycles;
    long long cycleLeadPrev = 0;
    //cycleLeadPrev is where the last iteration started looking at, cycleLeadNow is where this one started looking at, and cycleLeadNext is where we're looking at now
    long long *cycleLeadBounds;
    cudaMallocManaged(&cycleLeadBounds, sizeof(long long)*3);
    bool goForever = false; 
    if (repeatDepth == -1) {goForever = true; repeatDepth = 25;}
    for (int iters = 0; iters < repeatDepth; iters++) {
        for (int i = cycleLeadPrev; i < cycleLeadNow; i++) {
            //This is pretty inefficient since we're blatantly checking if any numbers are double counted but that also doesn't matter
            if (std::find(cycleLeads, cycleLeads + cycleLeadNext, ((cycleLeads[i] - meshOffset) << 1))==cycleLeads + cycleLeadNext) {
                //This part checks for the reverse of x/2
                cycleLeads[cycleLeadNext] = (cycleLeads[i] - meshOffset) << 1; cycleLeadNext++;
            }
            if (((cycleLeads[i] % 3 + 3) % 3 == ((2 + meshOffset) % 3 + 3) % 3) && (std::find(cycleLeads, cycleLeads + cycleLeadNext, (2 * (cycleLeads[i] - meshOffset)-1)/3)==cycleLeads + cycleLeadNext)) {
                //This part checks for the reverse of (3x+1)/2
                cycleLeads[cycleLeadNext] = (2 * (cycleLeads[i] - meshOffset) - 1) / 3; cycleLeadNext++;
            }
        }
        cycleLeadPrev = cycleLeadNow;
        cycleLeadNow = cycleLeadNext;
        if (cycleLeadNow > 2048 && goForever) {return new CollatzResults(0, iters);} //special result for benchmarking
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
            //basically a CPU version of the regular algorithm
            if (cycleStartMin <= x && x <= cycleStartMax) {
                int low = 0;
                int high = numCycles;  
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
                x *= 3; x++; x >>= 1; x += meshOffset;//(3x+1)/2+m
            } else {
                x >>= 1; x += meshOffset;//Still dividing by 2; right shift is 2% faster
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
                x *= 3; ++x; x >>= 1; x += meshOffset; ++t[i];
            } else {
                x >>= 1; x += meshOffset;
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
    if (65536 <= cycleLeadNow) {
        return new CollatzResults(-1, 0); //error result
    }
    if (2048 >= cycleLeadNow) {
        sharedMemLevel = 1;
    }
    if (512 >= cycleLeadNow && repeatDepth <= 11) {
        sharedMemLevel = 2;
    }

    //Enter hyperspeed
    cudaEventRecord(start);
    meshColGPURepeatedSteps<<<blocks, threads>>>(repeatDepth, sharedMemLevel, 1<<repeatDepth, c, s, t, cycleLeads, cycleLeadBounds, cycleLeadDelays, minN, maxN);
    cudaEventRecord(stop);


    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    
    unsigned long long delayTot = c[0];
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

    return new CollatzResults(milliseconds, delayTot);
}

/*
 * This is the kernel for running a Mesh-Collatz sequence with the traditional algorithm.
 * Arguments:
 * c <array of unsigned longs> - a blank array to add the delays of each thread to after program execution
 * minN <long> - the smallest value of |n| to test
 * maxN <long> - the largest value of |n| to test
 * meshOffset <int> - the Mesh offset to test
 * cycleStarts <array of longs> - a SORTED list of numbers that are the low points of a cycle
 * cycleStartSize <int> - the number of elements in cycleStarts (sizeof is weird)
 */
__global__
void meshColGPUShortcut(unsigned long long *c, long long minN, long long maxN, int meshOffset, long long *cycleStarts, int cycleStartSize) {
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
 * This is the function for running a Mesh-Collatz sequence with the traditional algorithm.
 * Arguments:
 * minN <long> - the number to start testing with (by absolute value)
 * maxN <long> - the number to test up to (by absolute value)
 * meshOffset <int> - the Mesh offset
 * blocks <int> - the number of CUDA blocks to use
 * threads <int> - the number of CUDA threads to use in each block
 */
CollatzResults* meshColShortcut(long long minN, long long maxN, int meshOffset, int blocks, int threads) {
    std::cout << "Testing up to n = +/- " << std::to_string(maxN) << std::endl; //Print testing range - again, this is just for testing purposes
    cudaEvent_t start, stop; //Add CUDA events for benchmarking
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    unsigned long long *c; //Initializing our c array
    cudaMallocManaged(&c, sizeof(long long)*blocks*threads);
    for (int i = 0; i < blocks*threads; i++) {
        c[i] = 0;
    }

    long long *cycleStarts;
    cudaMallocManaged(&cycleStarts, sizeof(long long)*10000);
    cycleStarts = meshColCycleSearch(100000+100000*(meshOffset > 0 ? meshOffset : -meshOffset), blocks, threads, meshOffset);
    int numCycles = cycleStarts[0];
    for (int i = 0; i < numCycles; i++) {
        cycleStarts[i] = cycleStarts[i+1];
    }

    cudaEventRecord(start);
    meshColGPUShortcut<<<blocks, threads>>>(c, minN, maxN, meshOffset, cycleStarts, numCycles);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();
    cudaEventSynchronize(stop); //Wait for everything to finish

    unsigned long long delayTot = c[0];
    for (int i = 1; i < blocks*threads; i++) {
        delayTot += c[i]; //Add up the c array to find the delay total
    }
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Elapsed time: " << std::to_string(milliseconds) << "ms" << std::endl;
    std::cout << "Tot delay: " << std::to_string(delayTot) << std::endl; //Sometimes there's a mismatch because of 64-bit overflow here but not in the more complex algorithm
    
    
    cudaFree(c);
    cudaFree(cycleStarts); //Free memory
    return new CollatzResults(milliseconds, delayTot);
}

/*
 * This is the function for finding the best sieve size for a specific Mesh offset.
 * Arguments:
 * minN <long> - the number to start testing with (by absolute value)
 * maxN <long> - the number to test up to (by absolute value)
 * meshOffset <int> - the Mesh offset
 * blocks <int> - the number of CUDA blocks to use
 * threads <int> - the number of CUDA threads to use in each block
 */
int meshColBench(int meshOffset, long long minN, long long maxN, int blocks, int threads) {
    //Benchmarking is now heuristic based, and this enables a flag that returns the largest result with <2048 cycle leads.
    CollatzResults* colres = meshColRepeatedSteps(-1, minN, maxN, meshOffset, blocks, threads);
    return colres->delay;
}
/*
 * This is the function that runs a Mesh offset for a sustained period of time, with automatic benchmarking and saving.
 * Arguments:
 * minN <long> - the number to start testing with (by absolute value)
 * maxN <long> - the number to test up to (by absolute value)
 * meshOffset <int> - the Mesh offset
 * maxM <int> - max Mesh offset tested, for savefiles
 * blocks <int> - the number of CUDA blocks to use
 * threads <int> - the number of CUDA threads to use in each block
 * SAVE_INTERVAL <long> - how many N to test before saving to a file
 * file <string> - the file name to save results to
 */
CollatzResults* meshColRunner(int meshOffset, int maxM, long long minN, long long maxN, int blocks, int threads, const long long SAVE_INTERVAL, std::string file, unsigned long long partialDelay) {
    //Save interval default value is 2^35
    CollatzResults* colres = new CollatzResults(0, partialDelay);
    long long benchMin = maxN >> 1;
    long long benchMax = benchMin + (1LL << 26);
    printf("> Benchmarking sieve sizes...\n");
    int sievePow = meshColBench(meshOffset, benchMin, benchMax, blocks, threads);
    printf("> Using %s%s2^%i%s sieve.\n\n", ANSI::italic, ANSI::brightYellow, sievePow, ANSI::reset);
    //All of this is pasted from meshColRepeatedSteps
    //Initialize c and CUDA Events
    unsigned long long *c;
    cudaMallocManaged(&c, sizeof(unsigned long long)*blocks*threads);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    //Find the numbers that go to 1 within repeatDepth steps.
    long long *cycleLeads;
    cudaMallocManaged(&cycleLeads, sizeof(long long)*2048);
    char *cycleLeadDelays;
    cudaMallocManaged(&cycleLeadDelays, 2048);
    printf("> Finding cycles...\n");
    const int CYCLE_SCALE = 100000;
    long long *cycleStarts2 = meshColCycleSearch(CYCLE_SCALE+CYCLE_SCALE*(meshOffset > 0 ? meshOffset : -meshOffset), blocks, threads, meshOffset);
    long long *cycleStarts;
    cudaMallocManaged(&cycleStarts, sizeof(long long)*2048);
    int numCycles = cycleStarts2[0];
    for (int i = 0; i < numCycles; i++) {
        cycleStarts[i] = cycleStarts2[i+1];
        cycleLeads[i] = cycleStarts[i];
    }
    long long cycleStartMin = cycleStarts[0];
    long long cycleStartMax = cycleStarts[numCycles-1];
    printf("> Found %s%s%i%s cycles, ranging from %lld - %lld.\n\n", ANSI::italic, ANSI::brightYellow, numCycles, ANSI::reset, cycleStartMin, cycleStartMax);
    //Save to file
    std::ofstream cycleFile(file+"_cycles_"+std::to_string(meshOffset)+".txt");
    cycleFile << "Total Cycles: " << numCycles << std::endl;
    cycleFile << cycleStarts[0];
    for (int i = 1; i < numCycles; i++) {
        cycleFile << "," << cycleStarts[i];
    }
    cycleFile.close();
    long long cycleLeadNext = numCycles;
    long long cycleLeadNow = numCycles;
    long long cycleLeadPrev = 0;
    //cycleLeadPrev is where the last iteration started looking at, cycleLeadNow is where this one started looking at, and cycleLeadNext is where we're looking at now
    long long *cycleLeadBounds;
    cudaMallocManaged(&cycleLeadBounds, sizeof(long long)*3);
    printf("> Generating cycle leads...\n");
    for (int iters = 0; iters < sievePow; iters++) {
        for (int i = cycleLeadPrev; i < cycleLeadNow; i++) {
            //This is pretty inefficient since we're blatantly checking if any numbers are double counted but that also doesn't matter
            if (std::find(cycleLeads, cycleLeads + cycleLeadNext, ((cycleLeads[i] - meshOffset) << 1))==cycleLeads + cycleLeadNext) {
                //This part checks for the reverse of x/2
                cycleLeads[cycleLeadNext] = (cycleLeads[i] - meshOffset) << 1; cycleLeadNext++;
            }
            if (((cycleLeads[i] % 3 + 3) % 3 == ((2 + meshOffset) % 3 + 3) % 3) && (std::find(cycleLeads, cycleLeads + cycleLeadNext, (2 * (cycleLeads[i] - meshOffset)-1)/3)==cycleLeads + cycleLeadNext)) {
                //This part checks for the reverse of (3x+1)/2
                cycleLeads[cycleLeadNext] = (2 * (cycleLeads[i] - meshOffset) - 1) / 3; cycleLeadNext++;
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
            //basically a CPU version of the regular algorithm
            if (cycleStartMin <= x && x <= cycleStartMax) {
                int low = 0;
                int high = numCycles;  
                while (low != high) {
                    int mid = (low+high)/2;
                    if (x == cycleStarts[mid]) {
                        goto endOfLoopShortcutInCPU2;
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
                x *= 3; x++; x >>= 1; x += meshOffset;//(3x+1)/2+m
            } else {
                x >>= 1; x += meshOffset;//Still dividing by 2; right shift is 2% faster
            }
        }
        endOfLoopShortcutInCPU2:;
    }
    printf("> %s%s%lld%s cycle leads generated, ranging from %lld - %lld.\n\n", ANSI::italic, ANSI::brightYellow, cycleLeadNow, ANSI::reset, cycleLeads[0], cycleLeads[cycleLeadNow - 1]);

    printf("> Generating sieve...\n");
    //Now we generate the sieve.
    //S saves the trajectory, T saves the number of 3x+1 steps. See the kernel for the theory, but we can use these to skip repeatDepth steps at a time.
    long long *s;
    char *t; //We're using char because we know t <= repeatDepth and repeatDepth <= 255
    cudaMallocManaged(&s, sizeof(long long)<<sievePow);
    cudaMallocManaged(&t, sizeof(char)<<sievePow);
    //This could be done on GPU, but if we're sticking with a 2^16 sieve it might not be worth it
    for (long long i = 0; i < 1<<sievePow; i++) {
        long long x = i;
        for (int j = 0; j < sievePow; j++) {
            if (x % 2) {
                x *= 3; ++x; x >>= 1; x += meshOffset; ++t[i];
            } else {
                x >>= 1; x += meshOffset;
            }
        }
        s[i] = x;
    }
    //Here's the logic for all of the shared memory things.
    int sharedMemLevel = 0;
    if (65536 <= cycleLeadNow) {
        return new CollatzResults(-1, 0); //error result
    }
    if (2048 >= cycleLeadNow) {
        sharedMemLevel = 1;
    }
    if (512 >= cycleLeadNow && sievePow <= 11) {
        sharedMemLevel = 2;
    }
    printf("> Sieve generated, starting GPU cycle searching...\n\n");
    for (long long w = minN; w <= maxN; w += SAVE_INTERVAL) {
        if (w != minN) { //Print totalDelay after a save but not at the end!
            printf("    Total Delay: %s%s%llu%s\n", ANSI::brightYellow, ANSI::italic, colres->delay, ANSI::reset);
        }
        
        //In preparation for the GPU, we initialize c[i];
        for (int i = 0; i < blocks*threads; i++) {
            c[i] = 0;
        }

        long long segmentMaxN = (w + SAVE_INTERVAL - 1 >= maxN) ? maxN : (w + SAVE_INTERVAL - 1);

        //Enter hyperspeed
        cudaEventRecord(start);
        meshColGPURepeatedSteps<<<blocks, threads>>>(sievePow, sharedMemLevel, 1<<sievePow, c, s, t, cycleLeads, cycleLeadBounds, cycleLeadDelays, w, segmentMaxN);
        cudaEventRecord(stop);


        cudaDeviceSynchronize();
        cudaEventSynchronize(stop);

        unsigned long long delayTot = c[0];
        for (int i = 1; i < blocks*threads; i++) {
            delayTot += c[i]; //Add up the c array to find the total delay
        }
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        colres->delay += delayTot;
        colres->milliseconds += milliseconds/1000;
        //Save to file
        std::ofstream savefile(file+"_checkpoint.txt");
        savefile << "minN=" << minN << "\nmaxN=" << maxN << "\nn=" << (segmentMaxN + 1) << "\nm=" << meshOffset << "\nmaxM=" << maxM << "\nsaveInterval=" << SAVE_INTERVAL << "\ndelay=" << colres->delay;
        savefile.close();
        std::ofstream partialdelayfile(file+"_partialDelaySums_"+std::to_string(meshOffset)+".txt", std::iostream::app);
        partialdelayfile << w << "," << segmentMaxN << "," << delayTot << "\n";
        partialdelayfile.close();
        printf("Checkpoint at n = %s%lld%s:\n    Time Taken: %s%fs%s\n    Delay: %s%llu%s\n", ANSI::underline, segmentMaxN, ANSI::reset, ANSI::underline, milliseconds/1000, ANSI::reset, ANSI::underline, delayTot, ANSI::reset);
    }
    
    std::cout << std::endl <<  "Elapsed time: " << std::to_string(colres->milliseconds) << "s" << std::endl;
    std::cout << "Total delay: " << ANSI::brightGreen << std::to_string(colres->delay) << ANSI::reset << std::endl;
    std::cout << "Average delay: " << ANSI::brightGreen << ANSI::italic << ANSI::underline << std::to_string((double)colres->delay/(2*(maxN-minN+1)-(minN == 0 ? 1 : 0))) << ANSI::reset << std::endl << std::endl;
    std::ofstream resfile(file+"_"+std::to_string(meshOffset)+".txt");
    resfile << "Delay: " << colres->delay << "\nAverage Delay: " << std::to_string((double)colres->delay/(2*(maxN-minN+1)-(minN == 0 ? 1 : 0))) << "\nTime: " << colres->milliseconds;
    resfile.close();
    
    //Make sure to free all the memory at the end
    cudaFree(c);
    cudaFree(s);
    cudaFree(t);
    cudaFree(cycleLeads);
    cudaFree(cycleStarts);
    cudaFree(cycleLeadBounds);
    cudaFree(cycleLeadDelays);

    return colres;
}

/*
 * This is the function that runs a Mesh offset for a sustained period of time, with automatic benchmarking and saving.
 * Arguments:
 * minN <long> - the number to start testing with (by absolute value)
 * maxN <long> - the number to test up to (by absolute value)
 * meshOffset <int> - the Mesh offset
 * blocks <int> - the number of CUDA blocks to use
 * threads <int> - the number of CUDA threads to use in each block
 * SAVE_INTERVAL <long> - how many N to test before saving to a file
 * file <string> - the file name to save results to
 * curN <long> - the current number being tested
 * partialDelay <unsigned long> - how much of the delay is calculated so far
 */
void meshColSearcher(int meshMin, int meshMax, long long minN, long long maxN, int blocks, int threads, const long long SAVE_INTERVAL, std::string file, long long curN, unsigned long long partialDelay) {
    int m = meshMin;
    while (true) {
        memcheck();
        printf("Testing m = %s%i%s...\n", ANSI::brightYellow, m, ANSI::reset);
        meshColRunner(m, meshMax, curN, maxN, blocks, threads, SAVE_INTERVAL, file, partialDelay);
        curN = minN;
        if (m == meshMax) exit(0);
        if (m >= 0) {
            m *= -1; m--;
        } else {
            m *= -1;
        }
    }
}

int main(int argc, char* argv[]) {
    //Right now this just runs a benchmark of all numbers up to +/- 10B
    ANSI::EnableVTMode();
    printf("%s%s%sMesh-Collatz Searcher v1.1.1%s\n\n", ANSI::italic, ANSI::underline, ANSI::brightBlue, ANSI::reset);
    //Initializing some vars to be taken in by the arguments
    std::string executionType = "";
    int meshMin = 0;
    int meshMax = 2147483647;
    bool largeCyclesOnly = false;
    long long minTestVal = -1;
    long long maxTestVal = -1;
    int sieveSize = 16;
    int threads = 512;
    int blocks = 512;
    long long saveInterval = 1LL << 35;
    std::string filename = "collatz";
    for (int i = 0; i < argc; i++) { //Argument interpreter - pretty self explanatory
        if (!strcmp(argv[i],"-h")) printf("%sHelp for Mesh-Collatz Searcher:%s\n\n-h  Prints this help command.\n-m  Minimum Mesh offset to test.\n-M  Maximum Mesh offset to test.\n-n  Minimum |n| to test.\n-N  Maximum |n| to test.\n-T  Number of threads to use on the GPU. Supported by everything.\n-B  Number of blocks to use on the GPU. Supported by everything.\n\n-b  Benchmark a Mesh offset. Supports -m, -n, and -N.\n\n-c  Test for cycles. Supports -m, -M, -N, and -L.\n-L  Only test for large values of cycles.\n\n-t  Benchmark the delay-testing portion of the program. Supports -m, -n, -N, -i, and -S.\n-s  Run the delay-testing portion of the program. Supports -m, -M, -n, -N, -i, and -f.\n\n-i  Interval for filesaving with -t and -s.\n-S  Sieve depth for delay testing.\n-f  Save to a specific file name.\n-F  Continue running a search (-s) from a file.\n\n--showMem  Shows information about remaining free memory.", ANSI::underline, ANSI::reset);
        if (!strcmp(argv[i],"-c")) {
            executionType = "Cycle";
            if (minTestVal == -1) minTestVal = 0;
            if (maxTestVal == -1) maxTestVal = 100000;
        }
        if (!strcmp(argv[i],"-L")) largeCyclesOnly = true;
        if (!strcmp(argv[i],"-t")) {
            executionType = "Test";
            if (minTestVal == -1) minTestVal = 0;
            if (maxTestVal == -1) maxTestVal = 10000000000LL;
            if (saveInterval == 1LL << 35) saveInterval = 1LL << 32;
        }
        if (!strcmp(argv[i],"-b")) {
            executionType = "Bench";
            if (minTestVal == -1) minTestVal = 1LL << 42;
            if (maxTestVal == -1) maxTestVal = minTestVal + (1LL << 26);
        }
        if (!strcmp(argv[i],"-s")) {
            executionType = "Search";
            if (minTestVal == -1) minTestVal = 0;
            if (maxTestVal == -1) maxTestVal = 1LL << 43;
        }
        if (!strcmp(argv[i],"-F")) {
            executionType = "Search_File";
            filename = argv[i+1];
        }
        if (!strcmp(argv[i],"-i")) saveInterval = std::stoll(argv[i+1]);
        if (!strcmp(argv[i],"-f")) filename = argv[i+1];
        if (!strcmp(argv[i],"-m")) meshMin = std::stoi(argv[i+1]);
        if (!strcmp(argv[i],"-M")) meshMax = std::stoi(argv[i+1]);
        if (!strcmp(argv[i], "-N")) maxTestVal = std::stoll(argv[i+1]);
        if (!strcmp(argv[i], "-n")) minTestVal = std::stoll(argv[i+1]);
        if (!strcmp(argv[i], "-S")) sieveSize = std::stoi(argv[i+1]);
        if (!strcmp(argv[i], "-T")) threads = std::stoi(argv[i+1]);
        if (!strcmp(argv[i], "-B")) blocks = std::stoi(argv[i+1]);
        if (!strcmp(argv[i],"--showMem")) showMemory = true;
    }
    if (executionType == "Cycle") {
        //Initializing some more variables
        long long maxNumCycles = 0;
        long long maxCycleStart = 0;
        double maxCycleWeight = 0;
        int m = meshMin;
        while (true) {
            long long* results = new long long[1048576];
            results = meshColCycleSearch(maxTestVal+maxTestVal*(m > 0 ? m : -m), blocks, threads, m, largeCyclesOnly);
            long long numCycles = *results;
            if (numCycles > maxNumCycles) {
                printf("\nRecord number of cycles: %s%lld%s in m = %i ", ANSI::underline, numCycles, ANSI::reset, m);
                maxNumCycles = numCycles;
            }
            for (int j = 1; j <= numCycles; j++) {
                long long cycleVal = *(results+j); //*(results+i) is like results[i];
                double cycleWeight = (cycleVal - 2*m) / (double)(4*m+1); //magic formula
                if (std::abs(cycleVal) > maxCycleStart) {
                    printf("\nRecord cycle start: %s%lld%s in m = %i ", ANSI::underline, *(results+j), ANSI::reset, m);
                    maxCycleStart = std::abs(cycleVal);
                }
                if (std::abs(cycleWeight) > maxCycleWeight) {
                    printf("\nRecord cycle weight: %s%f%s in m = %i ", ANSI::underline, (*(results+j)-2*m)/(double)(4*m+1), ANSI::reset, m);
                    maxCycleWeight = std::abs(cycleWeight);
                }
            }
            if (m == meshMax) exit(0);
            if (m >= 0) {
                m *= -1; m--;
            } else {
                m *= -1;
            }
            delete [] results;
        }
    }
    if (executionType == "Test") {
        printf("%sNo Sieve%s, m = %i\n", ANSI::underline, ANSI::reset, meshMin);
        meshColShortcut(minTestVal, maxTestVal, meshMin, blocks, threads);
        printf("%s2^%i Sieve%s, m = %i\n", ANSI::underline, sieveSize, ANSI::reset, meshMin);
        meshColRepeatedSteps(sieveSize, minTestVal, maxTestVal, meshMin, blocks, threads);
    }
    if (executionType == "Bench") {
        int svSize = meshColBench(meshMin, minTestVal, maxTestVal, blocks, threads);
        printf("%s%sBest Sieve Size: %s2^%i%s", ANSI::italic, ANSI::brightGreen, ANSI::underline, svSize, ANSI::reset);
    }
    if (executionType == "Search") {
        meshColSearcher(meshMin, meshMax, minTestVal, maxTestVal, blocks, threads, saveInterval, filename, minTestVal, 0);
    }
    if (executionType == "Search_File") {
        std::string line;
        std::ifstream savefile(filename+"_checkpoint.txt");
        std::getline(savefile, line);
        minTestVal = std::stoll(line.substr(5));
        std::getline(savefile, line);
        maxTestVal = std::stoll(line.substr(5));
        std::getline(savefile, line);
        long long curTestVal = std::stoll(line.substr(2));
        std::getline(savefile, line);
        meshMin = std::stoi(line.substr(2));
        std::getline(savefile, line);
        meshMax = std::stoi(line.substr(5));
        std::getline(savefile, line);
        saveInterval = std::stoll(line.substr(13));
        std::getline(savefile, line);
        unsigned long long partialDelay = std::stoull(line.substr(6));

        meshColSearcher(meshMin, meshMax, minTestVal, maxTestVal, blocks, threads, saveInterval, filename, curTestVal, partialDelay);
    }
}
