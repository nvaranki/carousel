
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>

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
cudaError_t alloc( int** a, int** d, int** dev_a, int** dev_b, int** dev_c, int** dev_d, size_t size )
{
    cudaError_t cudaStatus;

    // Allocate host buffers for two vectors

    cudaStatus = cudaMallocHost( a, size * sizeof(int) );// in page-locked memory
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "alloc host A failed %u\n", cudaStatus);
        return cudaStatus;
    }

    cudaStatus = cudaMallocHost( d, size * sizeof(int) );// in page-locked memory
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "alloc host D failed %u\n", cudaStatus);
        return cudaStatus;
    }

    // Allocate GPU buffers for three vectors

    cudaStatus = cudaMalloc((void**)dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc device A failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc device B failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc device C failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)dev_d, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc device D failed!");
        return cudaStatus;
    }

    return cudaStatus;
}

__host__
void releaseOnDevice( int* dev_a, int* dev_b, int* dev_c, int* dev_d )
{
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(dev_d);
}

__host__
void releaseOnHost( int* hst_a, int* hst_d )
{
    cudaFreeHost(hst_a);
    cudaFreeHost(hst_d);
}

__host__
void releaseOnHost( int* hst_a, int* hst_d, int* tst_a, int* tst_b, int* tst_c, int* tst_d )
{
    cudaFreeHost(hst_a);
    cudaFreeHost(hst_d);
    delete[] tst_a;
    delete[] tst_b;
    delete[] tst_c;
    delete[] tst_d;
}

__host__
cudaError_t upload( const int* hst_b, int* dev_b, size_t size )
{
    cudaError_t cudaStatus;

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy( dev_b, hst_b, size * sizeof(int), cudaMemcpyHostToDevice );
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy B failed!\n");
        return cudaStatus;
    }
    return cudaStatus;
}

__host__
cudaError_t upload( const int* hst_a, int* dev_a, size_t size, cudaStream_t stream )
{
    cudaError_t cudaStatus;

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpyAsync( dev_a, hst_a, size * sizeof(int), cudaMemcpyHostToDevice, stream );
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy A failed!\n");
        return cudaStatus;
    }
    return cudaStatus;
}

__host__
cudaError_t download( 
    int* hst_a, int* hst_b, int* hst_c, int* hst_d, 
    int* dev_a, int* dev_b, int* dev_c, int* dev_d, size_t size )
{
    cudaError_t cudaStatus;

    // Copy vectors from GPU buffer to host memory.
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
    cudaStatus = cudaMemcpy( hst_c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost );
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy C failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMemcpy( hst_d, dev_d, size * sizeof(int), cudaMemcpyDeviceToHost );
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy D failed!");
        return cudaStatus;
    }
    return cudaStatus;
}

__host__
cudaError_t download( int* hst_a, int* dev_a, size_t size, cudaStream_t stream )
{
    cudaError_t cudaStatus;

    // Copy input vectors from GPU buffer to host memory.
    cudaStatus = cudaMemcpyAsync( hst_a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost, stream );
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy A failed!");
        return cudaStatus;
    }
    return cudaStatus;
}

__host__
void fill( int base, int* a, size_t size)
{
    for (int i = 0; i < size; i++)
        a[i] = base + i;
}

// Helper function for using CUDA to process vectors in parallel.
__host__
cudaError_t carousel( 
    int* hst_a, int* hst_d, 
    int* dev_a, int* dev_b, int* dev_c, int* dev_d, 
    const unsigned int size, const unsigned int ts, const unsigned int repeat )
{
    cudaError_t cudaStatus;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaStream_t sX, sY, sI, sO;
    cudaStreamCreate( &sX );
    cudaStreamCreate( &sY );
    cudaStreamCreate( &sI );
    cudaStreamCreate( &sO );

    cudaEventRecord(startEvent, 0);

    const unsigned int tse(std::max(1, std::min((int)ts, 1024)));
    dim3 dg(std::max(1, (int)(size/tse))), db(std::min((int)tse, (int)size));

    // Launch synchronized streams on the GPU with one thread for each element.
    for( unsigned int i = 0; i < repeat; i++ )
    {
        // I: ↓X fill upload₀            ↓X fill upload₁            ↓X fill upload₂           ...
        // X:                ↑I ↓Y kAdd₀                ↑I ↓Y kAdd₁                ↑I ↓Y kAdd₂
        // Y:                           ↓O ↑X kSub₀                ↓O ↑X kSub₁                ↓O ↑X kSub₂
        // O:                                      ↑Y download₀               ↑Y download₁               ↑Y download₂

        // I: get and upload next input
        fprintf(stdout, "push %d\n", i);
        cudaStreamSynchronize( sX );
        fprintf(stdout, "push %d (sX)\n", i);
        fill( i, hst_a, size ); //TODO get new input data
        upload( hst_a, dev_a, size, sI );

        // X: dev_c = dev_a + dev_b
        fprintf(stdout, "add  %d\n", i);
        cudaStreamSynchronize( sI );
        fprintf(stdout, "add  %d (sI)\n", i);
        cudaStreamSynchronize( sY );
        fprintf(stdout, "add  %d (sY)\n", i);
        kAdd <<<dg,db,0,sX>>> (dev_c, dev_a, dev_b); 
        
        // Y: dev_d = dev_c - dev_b
        fprintf(stdout, "sub  %d\n", i);
        cudaStreamSynchronize( sO );
        fprintf(stdout, "sub  %d (sO)\n", i);
        cudaStreamSynchronize( sX );
        fprintf(stdout, "sub  %d (sX)\n", i);
        kSub <<<dg,db,0,sY>>> (dev_d, dev_c, dev_b);
        
        // O: download last result
        fprintf(stdout, "pull %d\n", i);
        cudaStreamSynchronize( sY );
        fprintf(stdout, "pull %d (sY)\n", i);
        download( hst_d, dev_d, size, sO );
        //TODO signal out the hst_d is ready
        cudaStreamSynchronize(sO);
        const char* fmt = "Result: %c={%4d,%4d,%4d,%4d,%4d} - pull\n";
        printf(fmt, 'D', hst_d[0], hst_d[1], hst_d[2], hst_d[3], hst_d[4]);
    }
    cudaDeviceSynchronize(); // waits until all streams finished

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

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float ms; // elapsed time in milliseconds
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    fprintf(stderr, "Time for carousel execute #%d: %.3f us (%.3f us/cycle,%.1f ns/cell)\n", repeat, ms*1000., ms*1000./repeat, ms*1000000./ size /repeat );
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

Error:
    cudaStreamDestroy( sX );
    cudaStreamDestroy( sY );
    cudaStreamDestroy( sI );
    cudaStreamDestroy( sO );

    return cudaStatus;
}

__host__
int main()
{
    const int device = 0;
    const int arraySize = /*5;/*/ 3 * 64 * 32; // 3 blocks to compute on 64*32=2048 cores
    fprintf(stderr, "Array size: %d\n", arraySize);
    int* a; // input
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int* d; // output
    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;
    int* dev_d = nullptr;
    cudaError_t cudaStatus = cudaErrorUnknown;

    // Setup and prepare.
    cudaStatus = setup(device);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "setup device %d failed %u\n", device, cudaStatus);
        return 1;
    }

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);

    cudaStatus = alloc( &a, &d, &dev_a, &dev_b, &dev_c, &dev_d, arraySize );
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "alloc failed %u\n", cudaStatus);
        releaseOnDevice( dev_a, dev_b, dev_c, dev_d );
        releaseOnHost( a, d );
        return 2;
    }
    cudaStatus = upload( b, dev_b, arraySize );
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "upload failed %u\n", cudaStatus);
        releaseOnDevice( dev_a, dev_b, dev_c, dev_d );
        releaseOnHost( a, d );
        return 3;
    }

    // Add-then-subtract vectors in parallel.
    cudaStatus = carousel( a, d, dev_a, dev_b, dev_c, dev_d, arraySize, 1024, 100000 );
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "carousel failed %u\n", cudaStatus);
        releaseOnDevice( dev_a, dev_b, dev_c, dev_d );
        releaseOnHost( a, d );
        return 4;
    }

    int* tst_a = new int[arraySize];
    int* tst_b = new int[arraySize];
    int* tst_c = new int[arraySize];
    int* tst_d = new int[arraySize];
    cudaStatus = download( tst_a, tst_b, tst_c, tst_d, dev_a, dev_b, dev_c, dev_d, arraySize );
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "upload failed %u\n", cudaStatus);
        releaseOnDevice( dev_a, dev_b, dev_c, dev_d );
        releaseOnHost( a, d, tst_a, tst_b, tst_c, tst_d );
        return 5;
    }

    //TODO compare

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float ms; // elapsed time in milliseconds
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    fprintf(stderr, "Time for the test execute: %.3f ms\n", ms);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    const char* fmt = "Result: %c={%4d,%4d,%4d,%4d,%4d}\n";
    printf(fmt, 'A', tst_a[0], tst_a[1], tst_a[2], tst_a[3], tst_a[4]);
    printf(fmt, 'B', tst_b[0], tst_b[1], tst_b[2], tst_b[3], tst_b[4]);
    printf(fmt, 'C', tst_c[0], tst_c[1], tst_c[2], tst_c[3], tst_c[4]);
    printf(fmt, 'D', tst_d[0], tst_d[1], tst_d[2], tst_d[3], tst_d[4]);

    releaseOnDevice( dev_a, dev_b, dev_c, dev_d );
    releaseOnHost( a, d, tst_a, tst_b, tst_c, tst_d );

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        releaseOnDevice( dev_a, dev_b, dev_c, dev_d );
        releaseOnHost( a, d, tst_a, tst_b, tst_c, tst_d );
        return 6;
    }

    return 0;
}
