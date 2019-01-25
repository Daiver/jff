#include "minitest.h"

#include <stdio.h>
#include <iostream>

#include <Eigen/Core>

#include "bestthreshold.h"
#include "bestsplit.h"
#include "buildsubtree.h"
#include "common.h"
#include "gradhessgainfunctor.h"

class GainStruct
{
public:
    GainStruct():a(0), b(0){}
    GainStruct(float a):a(a), b(0){}
    GainStruct(float a, float b):a(a), b(b){}
    float a, b;

    float sum() const
    {
        return a + b;
    }

    GainStruct &operator += (const GainStruct &other)
    {
        this->a += other.a;
        this->b += other.b;
        return *this;
    }

    GainStruct &operator -= (const GainStruct &other)
    {
        this->a -= other.a;
        this->b -= other.b;
        return *this;
    }
    GainStruct operator - (const GainStruct &other) const
    {
        GainStruct tmp = *this;
        tmp -= other;
        return tmp;
    }
};

double fabs(const GainStruct a)
{
    return fabs(a.a) + fabs(a.b);
}

template<class Scalar>
Scalar squareLoss(const Scalar f, const Scalar y)
{
    auto tmp = f - y;
    return tmp * tmp;
}

template<class Scalar>
Scalar squareLossGrad(const Scalar f, const Scalar y)
{
    return (f - y);
}

template<class Scalar>
Scalar squareLossHess(const Scalar f, const Scalar y)
{
    return 1;
}

template<class Scalar>
Scalar negLogLoss(const Scalar f, const Scalar y)
{
    return y * log(1.0 + exp(-f)) + (1.0 - y) * log(1 + exp(f));
}

template<class Scalar>
Scalar sigmoid(const Scalar z)
{
    return 1.0/(1.0 + exp(-z));
}

template<class Scalar>
Scalar negLogLossGrad(const Scalar f, const Scalar y)
{
    //return y * (-sigmoid(-f)) + (1.0 - y) * sigmoid(f);
    return (sigmoid(f)) - y;
}

template<class Scalar>
Scalar negLogLossHess(const Scalar f, const Scalar y)
{
    //return (1.0 - 2*y) * (sigmoid(-f)*(1.0 - sigmoid(-f)));
    auto tmp = sigmoid(f);
    return std::max(0.0001, tmp * (1.0 - tmp));
}

template<class Scalar>
Scalar buildAverageLeaf(
        const Eigen::Matrix<Scalar, -1, 1> &vals, 
        const std::vector<int> &dataIndices)
{
    const Scalar normConst = 1.0/dataIndices.size();
    Scalar res = 0;
    for(int i = 0; i < dataIndices.size(); ++i)
        res += vals[dataIndices[i]];
    return res * normConst;
}

void testFindBestThr01()
{
    auto entrImp = [=](const GainStruct &s){
        float sum = s.a + s.b;
        GainStruct tmp(s.a/sum, s.b/sum);
        if (fabs(sum) < 0.00001)
            tmp = GainStruct(0.001, 0.001);

        auto terma = -tmp.a * log(tmp.a);
        auto termb = -tmp.b * log(tmp.b);
        if(fabs(tmp.a) < 0.0001)
            return termb;
        if(fabs(tmp.b) < 0.0001)
            return terma;
        return terma + termb;
    };
    auto gainFunc = [&](
            const GainStruct &Gall, 
            const GainStruct &Gl, 
            const GainStruct &Gr){
        return entrImp(Gall) - Gl.sum()/Gall.sum() * entrImp(Gl) - Gr.sum() / Gall.sum() * entrImp(Gr);
    };

    Eigen::Matrix<float, -1, 1> values(5);
    values << 1, //0
              2, //1
              4, //2
              3, //3
              5; //4

    Eigen::Matrix<GainStruct, -1, 1> gainValues(5);
    gainValues << GainStruct(0, 1),//0
                  GainStruct(1, 0),//1
                  GainStruct(1, 0),//2
                  GainStruct(1, 0),//3
                  GainStruct(1, 0);//4
    double gain;
    float thr;
    findBestThreshold(values, gainValues, gainFunc, gain, thr);
    ASSERT(fabs(thr - 1.5) < 0.0001);
}

void testFindBestSpltGB01()
{
    Eigen::MatrixXf data(5, 2);
    data << 0, 0,
            1, 0,
            6, 1,
            5, 3,
            1, 9;
    Eigen::VectorXf values(5);
    values << 0, 0, 2, 3, 4;

    Eigen::VectorXf answers = Eigen::VectorXf::Zero(values.rows());

    std::vector<int> featIndices = {0, 1};

    GradHessGainFunctor<float> gainFunc(0, 0);

    Eigen::Matrix<GradHess<float>, -1, 1> gradAndHessPerSample = 
        computeGradAndHessPerSample<float>(
            values, answers,
            squareLossGrad<float>, squareLossHess<float>);


    std::vector<int> dataIndices(data.rows());
    for(int i = 0; i < dataIndices.size(); ++i)
        dataIndices[i] = i;

    double gain;
    int featInd;
    float thr;
    findBestSplit(
            data, gradAndHessPerSample, dataIndices, featIndices,
            gainFunc,
            gain, featInd, thr);
    ASSERT(featInd == 1);
    ASSERT(fabs(thr - 6) < 0.00001);
}

void testFindBestSpltGB02()
{
    Eigen::MatrixXf data(5, 2);
    data << 0, 0,
            1, 0,
            6, 1,
            5, 3,
            1, 9;
    Eigen::VectorXf values(5);
    values << 0, 0, 2, 3, 4;

    Eigen::VectorXf answers = Eigen::VectorXf::Zero(values.rows());

    std::vector<int> featIndices = {0};

    Eigen::Matrix<GradHess<float>, -1, 1> gradAndHessPerSample = 
        computeGradAndHessPerSample<float>(
            values, answers,
            squareLossGrad<float>, squareLossHess<float>);

    GradHessGainFunctor<float> gainFunc(0, 0);

    std::vector<int> dataIndices(data.rows());
    for(int i = 0; i < dataIndices.size(); ++i)
        dataIndices[i] = i;

    double gain;
    int featInd;
    float thr;
    findBestSplit(
            data, gradAndHessPerSample, dataIndices, featIndices,
            gainFunc,
            gain, featInd, thr);
    ASSERT(featInd == 0);
}

void testFindBestSpltGB03()
{
    Eigen::MatrixXf data(5, 2);
    data << 0, 0,
            1, 0,
            6, 1,
            5, 3,
            1, 9;
    Eigen::VectorXf values(5);
    values << 0, 0, 2, 3, 4;

    Eigen::VectorXf answers = Eigen::VectorXf::Zero(values.rows());

    std::vector<int> featIndices = {0};

    Eigen::Matrix<GradHess<float>, -1, 1> gradAndHessPerSample = 
        computeGradAndHessPerSample<float>(values, answers, 
                squareLossGrad<float>, squareLossHess<float>);

    GradHessGainFunctor<float> gainFunc(0, 0);

    std::vector<int> dataIndices = {0, 2};

    double gain;
    int featInd;
    float thr;
    findBestSplit(
            data, gradAndHessPerSample, dataIndices, featIndices,
            gainFunc,
            gain, featInd, thr);
    ASSERT(featInd == 0);
    ASSERT(fabs(thr - 3.0) < 0.001);
}
void testBuildTree01()
{
    Eigen::MatrixXf data(5, 2);
    data << 0, 0,
            1, 0,
            6, 1,
            5, 3,
            1, 9;
    Eigen::VectorXf values(5);
    values << 0, 0, 2, 3, 4;

    Eigen::VectorXf answers = Eigen::VectorXf::Zero(values.rows());

    Eigen::Matrix<GradHess<float>, -1, 1> gradAndHessPerSample = 
        computeGradAndHessPerSample<float>(values, answers, 
                squareLossGrad<float>, squareLossHess<float>);

    std::vector<int> dataIndices(data.rows());
    for(int i = 0; i < dataIndices.size(); ++i)
        dataIndices[i] = i;

    GradHessGainFunctor<float> gainFunc(0, 0);

    Node<float, float> *node = buildSubTree<float, float>(
            data, values, gradAndHessPerSample,
            dataIndices,
            gainFunc, buildAverageLeaf<float>);
    for(int i = 0; i < data.rows(); ++i){
        Eigen::VectorXf sample = data.row(i);
        float label = values[i];
        float ans = node->predict(sample);
        ASSERT(fabs(ans - label) < 0.0001);
        //std::cout << "pred " << ans << " label " << label << std::endl;
    }
}

void testBuildTree02()
{
    Eigen::MatrixXf data(5, 2);
    data << 0, 0,
            1, 0,
            6, 1,
            5, 3,
            1, 9;
    Eigen::VectorXf values(5);
    values << 0, 0, 1, 1, 1;

    Eigen::VectorXf answers = Eigen::VectorXf::Zero(values.rows());

    std::vector<int> dataIndices(data.rows());
    for(int i = 0; i < dataIndices.size(); ++i)
        dataIndices[i] = i;

    Eigen::Matrix<GradHess<float>, -1, 1> gradAndHessPerSample = 
        computeGradAndHessPerSample<float>(values, answers, 
                negLogLossGrad<float>, negLogLossHess<float>);

    GradHessGainFunctor<float> gainFunc(0, 0);

    Node<float, float> *node = buildSubTree<float, float>(
            data, values, gradAndHessPerSample,
            dataIndices,
            gainFunc, buildAverageLeaf<float>, 
            -1, 0, false);
    for(int i = 0; i < data.rows(); ++i){
        Eigen::VectorXf sample = data.row(i);
        float label = values[i];
        float ans = node->predict(sample);
        ASSERT(fabs(ans - label) < 0.0001);
        //std::cout << "pred " << ans << " label " << label << std::endl;
    }
}

void testBuildTree03()
{
    Eigen::MatrixXf data(2, 2);
    data << 0, 0,
            1, 0
            ;
    Eigen::VectorXf values(data.rows());
    values << 1, 1;

    Eigen::VectorXf answers = Eigen::VectorXf::Zero(values.rows());

    std::vector<int> dataIndices = {0, 1};
    //for(int i = 0; i < dataIndices.size(); ++i)
        //dataIndices[i] = i;

    Eigen::Matrix<GradHess<float>, -1, 1> gradAndHessPerSample = 
        computeGradAndHessPerSample<float>(values, answers, 
                negLogLossGrad<float>, negLogLossHess<float>);

    GradHessGainFunctor<float> gainFunc(0, 0);

    Node<float, float> *node = buildSubTree<float, float>(
            data, values, gradAndHessPerSample,
            dataIndices,
            gainFunc, buildAverageLeaf<float>, 
            -1, 0, true);
    for(int i = 0; i < data.rows(); ++i){
        Eigen::VectorXf sample = data.row(i);
        float label = values[i];
        float ans = node->predict(sample);
        //ASSERT(fabs(ans - label) < 0.0001);
        std::cout << "pred " << ans << " label " << label << std::endl;
    }
    //for(int i = 0; i < gradAndHessPerSample.rows(); ++i)
        //std::cout << gradAndHessPerSample[i].grad << " " 
                  //<< gradAndHessPerSample[i].hess << std::endl;
    node->print();
}


int main()
{
    RUN_TEST(testFindBestThr01);
    RUN_TEST(testFindBestSpltGB01);
    RUN_TEST(testFindBestSpltGB02);
    RUN_TEST(testFindBestSpltGB03);
    RUN_TEST(testBuildTree01);
    RUN_TEST(testBuildTree02);
    //RUN_TEST(testBuildTree03);
    return 0;
}
