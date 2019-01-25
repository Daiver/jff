#include "nodeleaf.h"

Trees::NodeLeaf::NodeLeaf(const float value): pValue(value)
{

}

float Trees::NodeLeaf::predict(const Eigen::VectorXf &) const
{
    return pValue;
}
