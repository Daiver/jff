#ifndef NODEBASE_H
#define NODEBASE_H

#include "Eigen/Core"

namespace Trees {

class NodeBase
{
public:
    virtual float predict(const Eigen::VectorXf &sample) = 0;
};

}

#endif // NODEBASE_H
