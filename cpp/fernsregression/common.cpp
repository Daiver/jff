#include "common.h"

std::vector<std::pair<float, float>> computeRangesOfFeatures(const DataSet &dataSet)
{
    std::vector<std::pair<float, float>> res(dataSet[0].size(), std::make_pair(FLT_MAX, FLT_MIN));

    for(int i = 0; i < dataSet.size(); ++i)
        for(int j = 0; j < dataSet[i].size(); ++j){
            if(dataSet[i][j] < res[j].first)
                res[j].first = dataSet[i][j];
            if(dataSet[i][j] > res[j].second)
                res[j].second = dataSet[i][j];
        }

    return res;
}

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}


