#ifndef TARGETLOSSINFOMSE_H
#define TARGETLOSSINFOMSE_H

#include <assert.h>
#include <vector>
#include "Eigen/Core"

namespace Stump {

class TargetsLossInfoMSE
{
public:
    TargetsLossInfoMSE() {}
    TargetsLossInfoMSE(const float sumOfItems, const float sumOfSquares, const int nItems);

    template<typename Derived>
    static TargetsLossInfoMSE fromTargets(const Eigen::MatrixBase<Derived> &targets, const std::vector<long> &indices);

    float mean() const;
    float loss() const;

    void removeItem(const float item);
    void addItem(const float item);

private:
    float pSumOfItems = 0;
    float pSumOfSquares = 0;
    int   pNItems = 0;
};

}






//IMPLEMENTATIONS


inline Stump::TargetsLossInfoMSE::TargetsLossInfoMSE(const float sumOfItems, const float sumOfSquares, const int nItems):
    pSumOfItems(sumOfItems), pSumOfSquares(sumOfSquares), pNItems(nItems) {}

inline float Stump::TargetsLossInfoMSE::mean() const
{
    assert(pNItems > 0);
    return pSumOfItems / pNItems;
}

inline float Stump::TargetsLossInfoMSE::loss() const
{
    const float nItems = pNItems;
    const float mean = this->mean();
    return 1.0/nItems * (pSumOfSquares - 2 * mean * pSumOfItems + nItems * mean * mean);
}


template<typename Derived>
inline Stump::TargetsLossInfoMSE Stump::TargetsLossInfoMSE::fromTargets(const Eigen::MatrixBase<Derived> &targets, const std::vector<long> &indices)
{

    int nItems = 0;
    float sumOfItems = 0;
    float sumOfSquares = 0;
    for(const int index : indices){
        sumOfItems += targets(index);
        sumOfSquares += targets(index)*targets(index);
        ++nItems;
    }

    return TargetsLossInfoMSE(sumOfItems, sumOfSquares, nItems);
}

inline void Stump::TargetsLossInfoMSE::removeItem(const float item)
{
    assert(pNItems > 0);
    --pNItems;
    pSumOfItems -= item;
    pSumOfSquares -= item * item;
}

inline void Stump::TargetsLossInfoMSE::addItem(const float item)
{
    ++pNItems;
    pSumOfItems += item;
    pSumOfSquares += item * item;
}

#endif // TARGETLOSSINFOMSE_H
