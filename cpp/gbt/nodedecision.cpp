#include "nodedecision.h"


float Trees::NodeDecision::predict(const Eigen::VectorXf &sample)
{
    if(sample[splittingInfo.featInd] < splittingInfo.threshold){
        return this->pLeft->predict(sample);
    }else{
        return this->pRight->predict(sample);
    }
}
