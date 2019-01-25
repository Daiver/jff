#include "common.h"

ml::DataSet ml::datasetFrom2DArray(const std::vector<std::vector<float> > &arr)
{
    DataSet dataset(arr.size(), NULL);
    for(int i = 0; i < arr.size(); ++i)
        dataset[i] = &arr[i];
    return dataset;
}

int ml::bestFreq(const std::vector<float> &freqs)
{
    int res = -1;
    float max = 0.0;
    for(int i = 0; i < freqs.size(); ++i)
        if(max < freqs[i]){
            max = freqs[i];
            res = i;
        }
    return res;
}

