
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ 
void kAdd(int *c, const int *a, const int *b)
{
    const int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ 
void kSub(int *c, const int *a, const int *b)
{
    const int i = threadIdx.x;
    c[i] = a[i] - b[i];
}

__host__
cudaError_t setup(int dn)
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(dn);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }
    return cudaStatus;
}

__host__
cudaError_t alloc( int** dev_a, int** dev_b, int** dev_c, size_t size )
{
    cudaError_t cudaStatus;

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }
    return cudaStatus;
}

__host__
void release( int* dev_a, int* dev_b, int* dev_c )
{
    cudaError_t cudaStatus;

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}

__host__
void release( int* dev_a, int* dev_b, int* dev_c, int* hst_a, int* hst_b, int* hst_c )
{
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    delete[] hst_a;
    delete[] hst_b;
    delete[] hst_c;
}

__host__
cudaError_t upload( const int* hst_a, const int* hst_b, int* dev_a, int* dev_b, size_t size )
{
    cudaError_t cudaStatus;

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy( dev_a, hst_a, size * sizeof(int), cudaMemcpyHostToDevice );
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy A failed!\n");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy( dev_b, hst_b, size * sizeof(int), cudaMemcpyHostToDevice );
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy B failed!\n");
        return cudaStatus;
    }
    return cudaStatus;
}

__host__
cudaError_t download( int* hst_a, int* hst_b, int* hst_c, int* dev_a, int* dev_b, int* dev_c, size_t size )
{
    cudaError_t cudaStatus;

    // Copy input vectors from GPU buffer to host memory.
    cudaStatus = cudaMemcpy( hst_a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost );
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy A failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMemcpy( hst_b, dev_b, size * sizeof(int), cudaMemcpyDeviceToHost );
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy B failed!");
        return cudaStatus;
    }
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy( hst_c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost );
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy C failed!");
        return cudaStatus;
    }
    return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
__host__
cudaError_t carousel( int* dev_a, int* dev_b, int* dev_c, unsigned int size, unsigned int repeat )
{
    cudaError_t cudaStatus;

    for( int i = 0; i < repeat; i++ )
    {
        // Launch a kernel on the GPU with one thread for each element.
        kAdd <<<1,size>>> (dev_c, dev_a, dev_b); // dev_c = dev_a + dev_b
        kSub <<<1,size>>> (dev_a, dev_c, dev_b); // dev_a = dev_c - dev_b
    }
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "kAdd-kSub launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kAdd!\n", cudaStatus);
        goto Error;
    }

    Error:
    return cudaStatus;
}

__host__
int main()
{
    const int device = 0;
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };
    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;
    cudaError_t cudaStatus = cudaErrorUnknown;
    
    printf("Source:\nA={%3d,%3d,%3d,%3d,%3d}\nB={%3d,%3d,%3d,%3d,%3d}\nC={%3d,%3d,%3d,%3d,%3d}\n",
        a[0], a[1], a[2], a[3], a[4],
        b[0], b[1], b[2], b[3], b[4],
        c[0], c[1], c[2], c[3], c[4]
    );

    // Setup and prepare.
    cudaStatus = setup(device);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "setup device %d failed %u\n", device, cudaStatus);
        return 1;
    }
    cudaStatus = alloc( &dev_a, &dev_b, &dev_c, arraySize );
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "alloc failed %u\n", cudaStatus);
        release( dev_a, dev_b, dev_c );
        return 2;
    }
    cudaStatus = upload( a, b, dev_a, dev_b, arraySize );
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "upload failed %u\n", cudaStatus);
        release( dev_a, dev_b, dev_c );
        return 3;
    }

    // Add-then-subtract vectors in parallel.
    cudaStatus = carousel( dev_a, dev_b, dev_c, arraySize, 1000 );
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "carousel failed %u\n", cudaStatus);
        return 4;
    }

    int* tst_a = new int[arraySize];
    int* tst_b = new int[arraySize];
    int* tst_c = new int[arraySize];
    cudaStatus = download( tst_a, tst_b, tst_c, dev_a, dev_b, dev_c, arraySize );
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "upload failed %u\n", cudaStatus);
        release( dev_a, dev_b, dev_c, tst_a, tst_b, tst_c );
        return 5;
    }

    //TODO compare

    printf("Result:\nA={%3d,%3d,%3d,%3d,%3d}\nB={%3d,%3d,%3d,%3d,%3d}\nC={%3d,%3d,%3d,%3d,%3d}\n",
        tst_a[0], tst_a[1], tst_a[2], tst_a[3], tst_a[4],
        tst_b[0], tst_b[1], tst_b[2], tst_b[3], tst_b[4],
        tst_c[0], tst_c[1], tst_c[2], tst_c[3], tst_c[4]
        );

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        release( dev_a, dev_b, dev_c, tst_a, tst_b, tst_c );
        return 6;
    }

    release( dev_a, dev_b, dev_c, tst_a, tst_b, tst_c );
    return 0;
}
