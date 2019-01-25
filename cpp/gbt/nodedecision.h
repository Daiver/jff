#ifndef NODEDECISION_H
#define NODEDECISION_H

#include <memory>
#include "nodebase.h"
#include "splittinginfo.h"

namespace Trees {

class NodeDecision : public NodeBase
{
public:
    float predict(const Eigen::VectorXf &sample);

private:
    Stump::SplittingInfo splittingInfo;

    std::shared_ptr<NodeBase> pLeft;
    std::shared_ptr<NodeBase> pRight;
};

}

#endif // NODEDECISION_H
