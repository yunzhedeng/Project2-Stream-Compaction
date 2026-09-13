CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 2**

* Yunzhe Deng
  * [LinkedIn](https://www.linkedin.com/in/yunzhedeng), [personal website](https://yunzhedeng.com)
* Tested on: Windows 11, Intel Core i7-10750H @ 2.60GHz, 16 GB RAM, NVIDIA GeForce RTX 2060 (Personal Computer)

## 1. Project Description

This project implements stream compaction and prefix sum (scan) algorithms on both the CPU and GPU using CUDA. Stream compaction is implemented using the map-scan-scatter approach to remove zero-valued elements from an input array.

Here are its features:

- Serial CPU exclusive scan
- CPU stream compaction without scan
- CPU stream compaction using map-scan-scatter
- Naive parallel GPU exclusive scan
- Work-efficient GPU exclusive scan
- Work-efficient GPU stream compaction
- Thrust exclusive scan
- Optimized active-thread scheduling during up-sweep and down-sweep
- GPU radix sort for non-negative integers
- Shared-memory work-efficient scan
- Shared-memory bank-conflict optimization
- Power-of-two and non-power-of-two input support

## 2. Extra Credits

### 2.1 Optimized Work-Efficient GPU Scan

The work-efficient scan uses fewer threads when fewer calculations are needed. The number of active threads changes at each level based on the current offset, which avoids launching many threads that would do nothing.

```cpp
int activeThreads = paddedN / (offset * 2);
int blocks = (activeThreads + blockSize - 1) / blockSize;
```

### 2.2 Radix Sort

Implemented a GPU radix sort for non-negative integers using the same map-scan-scatter idea from stream compaction. For each bit, the values with a 0 bit are moved to the front and the values with a 1 bit are moved after them. This process is repeated for each bit until the array is sorted. The value of radix sort is that it shows how scan can be reused as a building block for a more complex parallel algorithm.

Example call:

```cpp
StreamCompaction::Radix::sort(SIZE, output, input);
```

Example output:

![](./own-img/radix.png)

### 2.3 GPU Scan Using Shared Memory && Hardware Optimization

Implemented another work-efficient scan using shared memory instead of repeatedly accessing global memory during the up-sweep and down-sweep. Each thread loads two elements from global memory into shared memory. The scan is then performed inside shared memory, and the final result is written back to global memory. This reduces the number of global memory accesses during the scan.

Adding padding to the shared-memory indices to reduce shared-memory bank conflicts:

```cpp
int bankOffset = index >> 5;
temp[index + bankOffset]
```
