#ifndef RANDOMFERN_H
#define RANDOMFERN_H

#include "Eigen/Core"

namespace RandomFern {
template<int Depth, typename Output>
class Fern
{
public:
    Fern() {}

//private:

    template<typename Derived>
    inline int getIndex(const Eigen::MatrixBase<Derived> &vec) const
    {
        int binInd = 0;
        for(int i = 0; i < Depth; ++i){
            binInd *= 2;
            if(vec[featInds[i]] >= thresholds[i])
                binInd += 1;
        }
        return binInd;
    }

    inline Output predict(const Eigen::VectorXf &sample) const 
    {
        return bins[getIndex(sample)];
    }

    Eigen::Matrix<Output, -1, 1> predictMass(const Eigen::MatrixXf &data) const
    {
        Eigen::Matrix<Output, -1, 1> res(data.rows(), 1);
        for(int i = 0; i < res.rows(); ++i){
            res[i] = predict(data.row(i));
        }
        return res;
    }

    static const int nBins = 1 << Depth;
    Eigen::Matrix<float, Depth, 1> thresholds;
    Eigen::Matrix<int, Depth, 1> featInds;
    Eigen::Matrix<Output, nBins, 1> bins;
    //float thresholds[Depth];
    //float featInds[Depth];
    //Output bins[nBins] ;
};
}

#endif
