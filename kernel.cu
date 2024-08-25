
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
    unsigned int size, unsigned int repeat )
{
    cudaError_t cudaStatus;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);

    cudaStream_t sX, sY, sI, sO;
    cudaStreamCreate( &sX );
    cudaStreamCreate( &sY );
    cudaStreamCreate( &sI );
    cudaStreamCreate( &sO );

    // Launch synchronized streams on the GPU with one thread for each element.
    for( unsigned int i = 0; i < repeat; i++ )
    {
        // I:            ↓X fill upload            ↓X fill upload 
        // X: ↑I ↓Y kAdd                ↑I ↓Y kAdd
        // Y:            ↓O ↑X kSub                ↓O ↑X kSub 
        // O:                       ↑Y download               ↑Y download 

        // get and upload next input
        cudaStreamSynchronize( sX );
        fill( i, hst_a, size ); //TODO get new input data
        upload( hst_a, dev_a, size, sI );

        // dev_c = dev_a + dev_b
        cudaStreamSynchronize( sI );
        cudaStreamSynchronize( sY );
        kAdd <<<1,size,0,sX>>> (dev_c, dev_a, dev_b); 
        
        // dev_d = dev_c - dev_b
        cudaStreamSynchronize( sO );
        cudaStreamSynchronize( sX );
        kSub <<<1,size,0,sY>>> (dev_d, dev_c, dev_b); 
        
        // download last result
        cudaStreamSynchronize( sY );
        download( hst_d, dev_d, size, sO );
        //TODO signal out the hst_d is ready
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

Error:
    cudaStreamDestroy( sX );
    cudaStreamDestroy( sY );
    cudaStreamDestroy( sI );
    cudaStreamDestroy( sO );

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float ms; // elapsed time in milliseconds
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    fprintf(stderr, "Time for carousel execute #%d: %.3f us (%.3f us/cycle)\n", repeat, ms*1000., ms*1000./repeat );
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return cudaStatus;
}

__host__
int main()
{
    const int device = 0;
    const int arraySize = 5;
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
    cudaStatus = carousel( a, d, dev_a, dev_b, dev_c, dev_d, arraySize, 1000 );
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
