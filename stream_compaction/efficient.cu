#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int offset, int* data) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;

            int index = (i + 1) * offset * 2 - 1;

            if (index < n) {
                data[index] += data[index - offset];
            }
        }

        __global__ void kernDownSweep(int n, int offset, int* data) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;

            int index = (i + 1) * offset * 2 - 1;

            if (index < n) {
                int temp = data[index - offset];

                data[index - offset] = data[index];

                data[index] += temp;
            }
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            if (n <= 0) {
                return;
            }

            int levels = ilog2ceil(n);

            int paddedN = 1 << levels;

            int* devData;

            cudaMalloc((void**)&devData, paddedN * sizeof(int));

            cudaMemset(devData, 0, paddedN * sizeof(int));

            cudaMemcpy(
                devData,
                idata,
                n * sizeof(int),
                cudaMemcpyHostToDevice
            );

            const int blockSize = 256;

            timer().startGpuTimer();
            // TODO
            for (int d = 0; d < levels; d++) {

                int offset = 1 << d;

                int activeThreads = paddedN / (offset * 2);

                int blocks = (activeThreads + blockSize - 1) / blockSize;

                //UP-SWEEP
                kernUpSweep<<<blocks, blockSize>>>(
                    paddedN,
                    offset,
                    devData
                );

                checkCUDAError("kernUpSweep failed");
            }
            
            cudaMemset(
                devData + paddedN - 1,
                0,
                sizeof(int)
            );

            //DOWN-SWEEP
            for (int d = levels - 1; d >= 0; d--) {

                int offset = 1 << d;

                int activeThreads = paddedN / (offset * 2);

                int blocks = (activeThreads + blockSize - 1) / blockSize;

                kernDownSweep<<<blocks, blockSize>>>(
                    paddedN,
                    offset,
                    devData
                );

                checkCUDAError("kernDownSweep failed");
            }
                        
            timer().endGpuTimer();

            cudaMemcpy(
                odata,
                devData,
                n * sizeof(int),
                cudaMemcpyDeviceToHost
            );

            cudaFree(devData);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {

            if (n <= 0) {
                return 0;
            }

            int levels = ilog2ceil(n);
            int paddedN = 1 << levels;

            const int blockSize = 128;
            int blocks = (n + blockSize - 1) / blockSize;

            int *devIdata;
            int *devOdata;
            int *devBools;
            int *devIndices;

            cudaMalloc((void**)&devIdata, n * sizeof(int));
            cudaMalloc((void**)&devOdata, n * sizeof(int));
            cudaMalloc((void**)&devBools, n * sizeof(int));
            cudaMalloc((void**)&devIndices, paddedN * sizeof(int));

            cudaMemcpy(
                devIdata,
                idata,
                n * sizeof(int),
                cudaMemcpyHostToDevice
            );

        
            cudaMemset(
                devIndices,
                0,
                paddedN * sizeof(int)
            );

            timer().startGpuTimer();
            // TODO
            
            // 1. Map
            Common::kernMapToBoolean<<<blocks, blockSize>>>(
                n,
                devBools,
                devIdata
            );

            checkCUDAError("kernMapToBoolean failed");

            cudaMemcpy(
                devIndices,
                devBools,
                n * sizeof(int),
                cudaMemcpyDeviceToDevice
            );

            //2. Scan

            // Up-Sweep
            for (int d = 0; d < levels; d++) {

                int offset = 1 << d;

                int activeThreads =
                    paddedN / (offset * 2);

                int scanBlocks =
                    (activeThreads + blockSize - 1) / blockSize;

                kernUpSweep<<<scanBlocks, blockSize>>>(
                    paddedN,
                    offset,
                    devIndices
                );

                checkCUDAError("compact up-sweep failed");
            }

            cudaMemset(
                devIndices + paddedN - 1,
                0,
                sizeof(int)
            );

            // Down-Sweep
            for (int d = levels - 1; d >= 0; d--) {

                int offset = 1 << d;

                int activeThreads =
                    paddedN / (offset * 2);

                int scanBlocks =
                    (activeThreads + blockSize - 1) / blockSize;

                kernDownSweep<<<scanBlocks, blockSize>>>(
                    paddedN,
                    offset,
                    devIndices
                );

                checkCUDAError("compact down-sweep failed");
            }

            // 3. Scatter
            Common::kernScatter<<<blocks, blockSize>>>(
                n,
                devOdata,
                devIdata,
                devBools,
                devIndices
            );

            checkCUDAError("kernScatter failed");

            timer().endGpuTimer();

            int lastIndex;
            int lastBool;

            cudaMemcpy(
                &lastIndex,
                devIndices + n - 1,
                sizeof(int),
                cudaMemcpyDeviceToHost
            );

            cudaMemcpy(
                &lastBool,
                devBools + n - 1,
                sizeof(int),
                cudaMemcpyDeviceToHost
            );

            int count = lastIndex + lastBool;

            if (count > 0) {
                cudaMemcpy(
                    odata,
                    devOdata,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                );
            }

            cudaFree(devIdata);
            cudaFree(devOdata);
            cudaFree(devBools);
            cudaFree(devIndices);

            return count;

        }
    }
}
