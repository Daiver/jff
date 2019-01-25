#include <vector>
#include <stdio.h>
#include <iostream>
#include <math.h>

template<class T> void printVector(std::vector<T> &vec)
{
    for(auto &x : vec)
        std::cout << x << " ";
    std::cout << std::endl;
}

float diffOfVectors(const std::vector<float> &a, const std::vector<float> &b)
{
    float err = 0;
    for(int i = 0; i < a.size(); ++i)
        err += (a[i] - b[i]) * (a[i] - b[i]);
    return err;
}

std::pair<std::vector<float>, float> linearRegression(
        const std::vector<std::vector<float>> &X,
        const std::vector<float> &Y,
        const std::vector<float> &initW,
        const float initB,
        const float lambda
        ){
    std::vector<float> W0 = initW;
    std::vector<float> W  = W0;

    const int numberOfFeatures = initW.size();
    const int numberOfSamples = X.size();

    float B0 = initB;
    float B  = B0;

    float err = 0;
    for(int iter = 0; iter < 100; ++iter){
        for(int j = 0; j < numberOfFeatures; ++j){
            float aii = lambda;
            for(int i = 0; i < numberOfSamples; ++i)
                aii += X[i][j] * X[i][j];
            float nominator = 0;
            for(int i = 0; i < numberOfSamples; ++i){
                for(int k = 0; k < numberOfFeatures; ++k){
                    if(k == j)
                        continue;
                    nominator += W0[k] * X[i][k] * X[i][j];
                }
                nominator += (B0 - Y[i]) * X[i][j];
            }
            W[j] = -nominator/aii;
        }
        float denom = numberOfSamples;
        float nominator = 0;
        for(int i = 0; i < numberOfSamples; ++i){
            for(int k = 0; k < numberOfFeatures; ++k)
                nominator += W0[k] * X[i][k];
            nominator -= Y[i];
        }
        B = -nominator/denom;
        err =  diffOfVectors(W, W0) + pow(B - B0, 2);
        //printf("%d err %f\n", iter, diffOfVectors(W, W0) + pow(B - B0, 2));
        //printf("%f |", B);
        //printVector(W);
        W0 = W;
        B0 = B;
    }
    printf("err %f\n", err);
    return std::make_pair(W, B);
}

float apply(const std::vector<float> &W, float &B, const std::vector<float> &x)
{
    float res = B;
    for(int i = 0; i < W.size(); ++i)
        res += W[i] * x[i];
    return res;
}

int main()
{
    std::vector<std::vector<float>> X = {
        {1.17},
        {2.97},
        {3.26},
        {4.69},
        {5.83},
        {6.41}
    };
    std::vector<float> Y = {
        78.93, 
        58.20, 
        67.47, 
        37.37,
        45.65,
        29.97};
    std::vector<float> initW = {0};
    auto res = linearRegression(X, Y, initW, 0, 0.0);
    printf("%f\n", res.second);
    printVector(res.first);
    printf("ans %f\n", apply(res.first, res.second, {5.83}));
    printf("ans %f\n", apply(res.first, res.second, {6.00}));
    printf("ans %f\n", apply(res.first, res.second, {6.41}));
    return 0;
}

/* 
1.17   78.93
2.97   58.20
3.26   67.47
4.69   37.47
5.83   45.65
6.00   32.92
6.41   29.97
 */
