#include <iostream>
#include <math.h>

__global__
void add(int n, float *x, float *y)
{
    for(int i = 0; i < n; ++i){
        y[i] = x[i] + y[i];
    }
}

int main(int, char **)
{
    const int N = 1 << 20;
    float *x;
    float *y;
    std::cout << "Before alloc" << std::endl;
    //x = new float[N];
    //y = new float[N];
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    std::cout << "Before fill" << std::endl;
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    return 0;

    std::cout << std::flush;
    std::cout << "Before add" << std::endl;
    std::cout << std::flush;
    add<<<1, 1>>>(N, x, y);

    std::cout << "Before sync" << std::endl;
    cudaDeviceSynchronize();
    std::cout << "Before err acc" << std::endl;
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    std::cout << "Before err free" << std::endl;
    cudaFree(x);
    cudaFree(y);

    return 0;
}

