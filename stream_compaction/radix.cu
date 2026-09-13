#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "radix.h"

namespace StreamCompaction {
    namespace Radix {

        using StreamCompaction::Common::PerformanceTimer;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernRadixUpSweep( int n, int offset,int* data) 
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;

            int index = (i + 1) * offset * 2 - 1;

            if (index < n) {
                data[index] += data[index - offset];
            }
        }

        __global__ void kernRadixDownSweep(int n, int offset, int* data
        ) 
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;

            int index = (i + 1) * offset * 2 - 1;

            if (index < n) {

                int temp = data[index - offset];

                data[index - offset] = data[index];

                data[index] += temp;
            }
        }

        // 1. Map
        __global__ void kernMapZeroBit(
            int n,
            int bit,
            int* zeroFlags,
            const int* data
        ) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index < n) {
                int currentBit = (data[index] >> bit) & 1;

                if (currentBit == 0) {
                    zeroFlags[index] = 1;
                }
                else {
                    zeroFlags[index] = 0;
                }
            }
        }

        // 2. Scatter
         __global__ void kernRadixScatter(
            int n,
            int* output,
            const int* input,
            const int* zeroFlags,
            const int* scanned
        ) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index < n) {

                int destination;

                int totalZeros = scanned[n - 1] + zeroFlags[n - 1];

                if (zeroFlags[index] == 1) {
                    destination = scanned[index];
                }
                else {
                    int onesBefore = index - scanned[index];

                    destination = totalZeros + onesBefore;
                }

                output[destination] = input[index];
            }
        }

        //3. Radix Sort
        void sort( int n, int* odata, const int* idata) 
        {
            if (n <= 0) {
                return;
            }

            int levels = ilog2ceil(n);
            int paddedN = 1 << levels;

            const int blockSize = 128;

            int blocks = (n + blockSize - 1) / blockSize;

            int* devA;
            int* devB;

            int* devZeroFlags;
            int* devScan;

            int* devTotalZeros;

            cudaMalloc((void**)&devA, n * sizeof(int));

            cudaMalloc((void**)&devB, n * sizeof(int));

            cudaMalloc((void**)&devZeroFlags, paddedN * sizeof(int));

            cudaMalloc((void**)&devScan, paddedN * sizeof(int));

            cudaMalloc((void**)&devTotalZeros,sizeof(int));

            cudaMemcpy(
                devA,
                idata,
                n * sizeof(int),
                cudaMemcpyHostToDevice
            );


            int* src = devA;
            int* dst = devB;


            timer().startGpuTimer();

            for (int bit = 0; bit < 31; bit++) {


                // 1. Map
                cudaMemset(
                    devZeroFlags,
                    0,
                    paddedN * sizeof(int)
                );

                kernMapZeroBit<<<blocks, blockSize>>>(
                    n,
                    bit,
                    devZeroFlags,
                    src
                );

                checkCUDAError("kernMapZeroBit failed");


                // 2. Scan
                cudaMemcpy(
                    devScan,
                    devZeroFlags,
                    paddedN * sizeof(int),
                    cudaMemcpyDeviceToDevice
                );

                // Up Sweep
                for (int d = 0; d < levels; d++) {

                    int offset = 1 << d;

                    int activeThreads = paddedN / (offset * 2);

                    int scanBlocks = (activeThreads + blockSize - 1) / blockSize;

                    kernRadixUpSweep<<<scanBlocks, blockSize>>>(
                        paddedN,
                        offset,
                        devScan
                    );

                    checkCUDAError("radix up-sweep failed");
                }


                // Exclusive scan:
                cudaMemset(
                    devScan + paddedN - 1,
                    0,
                    sizeof(int)
                );


                // Down Sweep
                for (int d = levels - 1; d >= 0; d--) {

                    int offset = 1 << d;

                    int activeThreads = paddedN / (offset * 2);

                    int scanBlocks = (activeThreads + blockSize - 1) / blockSize;

                    kernRadixDownSweep<<<scanBlocks, blockSize>>>(
                        paddedN,
                        offset,
                        devScan
                    );

                    checkCUDAError("radix down-sweep failed");
                }


                // 3. Scatter

                kernRadixScatter<<<blocks, blockSize>>>(
                    n,
                    dst,
                    src,
                    devZeroFlags,
                    devScan
                );

                checkCUDAError("kernRadixScatter failed");

                int* temp = src;
                src = dst;
                dst = temp;
            }


            timer().endGpuTimer();

            cudaMemcpy(
                odata,
                src,
                n * sizeof(int),
                cudaMemcpyDeviceToHost
            );

            cudaFree(devA);
            cudaFree(devB);
            cudaFree(devZeroFlags);
            cudaFree(devScan);
        }

    }
}