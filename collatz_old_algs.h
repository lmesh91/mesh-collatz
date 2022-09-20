__global__ void collatzGPURepeatedSteps(int repeatDepth, int sharedMemLevel, long long sieveSize, long long *c, long long *s, char *t, long long *cycleLeads, long long *cycleLeadBounds, char *cycleLeadDelays, long long minN, long long maxN);
float collatzRepeatedSteps(int repeatDepth, long long N, int blocks, int threads);
__global__ void collatzGPUShortcut(long long *c, long long minN, long  long maxN);
float collatzShortcut(long long N, int blocks, int threads);