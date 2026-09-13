#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernScan(int n, int offset,
                                 int *odata, const int *idata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index >= n) {
                return;
            }

            if (index >= offset) {
                odata[index] =
                    idata[index] + idata[index - offset];
            }
            else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int *devA;
            int *devB;

            cudaMalloc((void**)&devA, n * sizeof(int));
            cudaMalloc((void**)&devB, n * sizeof(int));

            cudaMemcpy(
                devA,
                idata,
                n * sizeof(int),
                cudaMemcpyHostToDevice
            );

            const int blockSize = 128;
            const int blocks = (n + blockSize - 1) / blockSize;
            
            timer().startGpuTimer();
            // TODO
            int *src = devA;
            int *dst = devB;

            for (int d = 0; d < ilog2ceil(n); d++) {

                int offset = 1 << d;

                kernScan<<<blocks, blockSize>>>(
                    n,
                    offset,
                    dst,
                    src
                );

                checkCUDAError("kernScan failed");

                int *temp = src;
                src = dst;
                dst = temp;
            }

            timer().endGpuTimer();

            odata[0] = 0;

            if (n > 1) {
                cudaMemcpy(
                    odata + 1,
                    src,
                    (n - 1) * sizeof(int),
                    cudaMemcpyDeviceToHost
                );
            }

            cudaFree(devA);
            cudaFree(devB);
        }
    }
}
