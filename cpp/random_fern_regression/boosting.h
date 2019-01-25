#ifndef BOOSTING_H
#define BOOSTING_H

#include <iostream>
#include <vector>
#include "ferntrain.h"
#include "cascade.h"

namespace RandomFern {

template<int Depth, typename Output, Output Gradient(const Output &)>
//template<int Depth, typename Output>
Cascade<Depth, Output> trainCascade(
	const Eigen::Matrix<std::pair<float, float>, -1, 1> &borders, 
	const Eigen::MatrixXf &data,
	const Eigen::Matrix<Output, -1, 1> &targetsInit,
	const int nRepeats,
	const int nStages,
    const float learningRate)
{
    typedef Fern<Depth, Output> FernRes;
	Cascade<Depth, Output> res;

    Eigen::Matrix<Output, -1, 1> targets       = targetsInit;
    Eigen::Matrix<Output, -1, 1> cascadeValues(targetsInit.rows());
    cascadeValues.fill(Output::Zero());

    for(int stageInd = 0; stageInd < nStages; ++stageInd){
        FernRes fern = trainFern<Depth, Output>(borders, data, targets, nRepeats);
        //const float weight = (stageInd == 0) ? 1.0 : learningRate;
        const float weight = learningRate;
        res.ferns.push_back(fern);
        res.weights.push_back(weight);
        for(int i = 0; i < data.rows(); ++i){
            //const Eigen::VectorXf sample = data.row(i);
            const Output prediction = fern.predict(data.row(i));
            cascadeValues[i]       += weight * prediction;
            const Output residual   = cascadeValues[i] - targetsInit[i];
            targets[i]              = -Gradient(residual);
        }
        //std::cout << stageInd << ": " << (res.predict(data) - targetsInit).norm() << std::endl;
    }

	return res;
}

}

#endif
