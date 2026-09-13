#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "shared.h"

namespace StreamCompaction {
    namespace Shared {

        using StreamCompaction::Common::PerformanceTimer;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __device__ int conflictFreeOffset(int index)
        {
            return index >> 5;
        }

        __global__ void kernSharedScan(int n, int paddedN, int* odata, const int* idata)
        {
            extern __shared__ int temp[];

            int tid = threadIdx.x;

            // One thread loads two elements
            int ai = tid;
            int bi = tid + blockDim.x;

            int bankOffsetA = conflictFreeOffset(ai);
            int bankOffsetB = conflictFreeOffset(bi);

            // Copy from global to shared
             if (ai < n) {
                temp[ai + bankOffsetA] = idata[ai];
            }
            else if (ai < paddedN) {
                temp[ai + bankOffsetA] = 0;
            }

            if (bi < n) {
                temp[bi + bankOffsetB] = idata[bi];
            }
            else if (bi < paddedN) {
                temp[bi + bankOffsetB] = 0;
            }

            __syncthreads();


            // Up-Sweep
            int offset = 1;

            for (int d = paddedN / 2; d > 0; d /= 2) {

                __syncthreads();

                if (tid < d) {
                    int left = offset * (2 * tid + 1) - 1;
                    int right = offset * (2 * tid + 2) - 1;

                    int leftOffset = conflictFreeOffset(left);
                    int rightOffset = conflictFreeOffset(right);

                    temp[right + rightOffset] += temp[left + leftOffset];
                }

                offset *= 2;
            }

            if (tid == 0) {
                int root = paddedN - 1;
                temp[root + conflictFreeOffset(root)] = 0;
            }


            // Down-Sweep
            for (int d = 1; d < paddedN; d *= 2) {

                offset /= 2;

                __syncthreads();

                if (tid < d) {
                    int left = offset * (2 * tid + 1) - 1;
                    int right = offset * (2 * tid + 2) - 1;

                    int leftOffset = conflictFreeOffset(left);
                    int rightOffset = conflictFreeOffset(right);

                    int t = temp[left + leftOffset];

                    temp[left + leftOffset] = temp[right + rightOffset];
                    temp[right + rightOffset] += t;
                }
            }

            __syncthreads();


            // Copy from shared to global
            if (ai < n) {
                odata[ai] = temp[ai + bankOffsetA];
            }

            if (bi < n) {
                odata[bi] = temp[bi + bankOffsetB];
            }
        }


        void scan(int n, int* odata, const int* idata)
        {
            if (n <= 0) {
                return;
            }

            if (n == 1) {
                odata[0] = 0;
                return;
            }

            int levels = ilog2ceil(n);
            int paddedN = 1 << levels;

            int threads = paddedN / 2;

            if (threads > 1024) {
                printf("Shared scan currently supports up to 2048 padded elements.\n");
                return;
            }

            int* devIdata;
            int* devOdata;

            cudaMalloc((void**)&devIdata, n * sizeof(int));
            cudaMalloc((void**)&devOdata, n * sizeof(int));

            cudaMemcpy(devIdata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int padding = paddedN >> 5;
            int sharedBytes = (paddedN + padding) * sizeof(int);

            timer().startGpuTimer();

            kernSharedScan<<<1, threads, sharedBytes>>>(n, paddedN, devOdata, devIdata);

            checkCUDAError("kernSharedScan failed");

            timer().endGpuTimer();

            cudaMemcpy(odata, devOdata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(devIdata);
            cudaFree(devOdata);
        }

    }
}