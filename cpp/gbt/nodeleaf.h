#ifndef NODELEAF_H
#define NODELEAF_H

#include "Eigen/Core"

namespace Trees {

class NodeLeaf
{
public:
    NodeLeaf() {}
    NodeLeaf(const float value);

    float predict(const Eigen::VectorXf &sample) const;

private:
    float pValue;

};

}
#endif // NODELEAF_H
