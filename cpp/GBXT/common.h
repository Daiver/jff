#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <Eigen/Core>

template<class Derived, class Scalar>
void findSplitIndices(
        const Eigen::MatrixBase<Derived> &data,
        const int featInd,
        const Scalar threshold,
        std::vector<int> &lInds,
        std::vector<int> &rInds)
{
    for(int i = 0; i < data.rows(); ++i)
        if(data(i, featInd) >= threshold)
            rInds.push_back(i);
        else
            lInds.push_back(i);
}

template<class Derived, class Scalar>
void findSplitIndices(
        const Eigen::MatrixBase<Derived> &data,
        const std::vector<int> &dataIndices,
        const int featInd,
        const Scalar threshold,
        std::vector<int> &lInds,
        std::vector<int> &rInds)
{
    for(int i = 0; i < dataIndices.size(); ++i)
        if(data(dataIndices[i], featInd) >= threshold)
            rInds.push_back(dataIndices[i]);
        else
            lInds.push_back(dataIndices[i]);
}


template<class Derived>
void splitMatrix(
        const Eigen::MatrixBase<Derived> &mat,
        const std::vector<int> &lInds,
        const std::vector<int> &rInds,
        Eigen::MatrixBase<Derived> &matL,
        Eigen::MatrixBase<Derived> &matR)
{
    for(int i = 0; i < lInds.size(); ++i)
        matL.row(i) = mat.row(lInds[i]);
    for(int i = 0; i < rInds.size(); ++i)
        matR.row(i) = mat.row(rInds[i]);
}



#endif
