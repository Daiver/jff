#ifndef EIGEXPORT_H
#define EIGEXPORT_H

#include <string>
#include <fstream>
#include <iostream>

#include "Eigen/Core"

namespace EigRoutine {

inline Eigen::MatrixXf readMatFromTxt(const std::string &fname)
{
    std::ifstream infile(fname);
    //infile.open()
    int rows = 0, cols = 0;
    infile >> rows >> cols;
    Eigen::MatrixXf res(rows, cols);
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            float value = 0;
            infile >> value;
            res(i, j) = value;
        }
    }
    return res;
}

}

#endif
