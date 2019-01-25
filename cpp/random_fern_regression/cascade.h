#ifndef CASCADE_H
#define CASCADE_H

#include <vector>
#include "fern.h"

namespace RandomFern {

template<int Depth, typename Output>
class Cascade {
public:
    template<typename Derived>
    inline Output predict(const Eigen::MatrixBase<Derived> &vec) const 
    {
        assert(ferns.size() > 0);
        Output res = weights[0] * ferns[0].predict(vec);
        for(int i = 1; i < ferns.size(); ++i){
            res += weights[i] * ferns[i].predict(vec);
        }
        return res;
    }

    Eigen::Matrix<Output, -1, 1> predictMass(const Eigen::MatrixXf &data) const
    {
        Eigen::Matrix<Output, -1, 1> res(data.rows());
        for(int i = 0; i < res.rows(); ++i){
            //const Eigen::VectorXf sample = data.row(i);
            res[i] = predict(data.row(i));
        }
        return res;
    }

    std::vector<Fern<Depth, Output> > ferns;
    std::vector<float> weights;
    
};

}

#endif
