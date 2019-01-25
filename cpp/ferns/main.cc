#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>
#include <eigen3/Eigen/Core>

#include "common.h"
#include "fern.h"

/*//namespace ml{*/

//typedef std::vector<float> FeatureVec;
//typedef std::vector<const FeatureVec *> DataSet;
/*}*/

int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

ml::DataSet readMNISTData(std::string fname)
{
    ml::DataSet res;
    std::ifstream in;
    in.open(fname.c_str(), std::ios::binary | std::ios::in);

    int magic;
    in.read((char*)&magic, sizeof(magic));
    magic = reverseInt(magic);
    if(magic != 2051){
        throw std::string("bad magic");
    }
    int countOfImages;
    in.read((char*)&countOfImages, sizeof(countOfImages));
    countOfImages = reverseInt(countOfImages);

    int rows;
    in.read((char*)&rows, sizeof(rows));
    rows = reverseInt(rows);

    int cols;
    in.read((char*)&cols, sizeof(cols));
    cols = reverseInt(cols);

    printf("images %d rows %d cols %d\n", countOfImages, rows, cols);

    res.resize(countOfImages, NULL);
    for(int i = 0; i < countOfImages; ++i){
        ml::FeatureVec *feats = new ml::FeatureVec(rows*cols, 0);
        for(int r = 0; r < rows; ++r){
            for(int c = 0; c < cols; ++c){
                unsigned char temp=0;
                in.read((char*)&temp,sizeof(temp));
                (*feats)[r*rows + c] = temp;
                //printf("%f\n", (float)temp);
            }
        }
        res[i] = feats;
    }

    in.close();
    return res;
}

std::vector<int> readMNISTLabels(std::string fname)
{
    std::ifstream in;
    in.open(fname.c_str(), std::ios::binary | std::ios::in);
    int magic;
    in.read((char*)&magic, sizeof(magic));
    magic = reverseInt(magic);

    int countOfLabels;
    in.read((char*)&countOfLabels, sizeof(countOfLabels));
    countOfLabels = reverseInt(countOfLabels);

    printf("magic %d %d\n", magic, countOfLabels);

    std::vector<int> res(countOfLabels, 0);
    unsigned char tmp;
    int counter = 0;
    while(in.good()){
        in.read((char*)&tmp, sizeof(tmp));
        res[counter] = tmp;
        counter++;
    }

    in.close();
    return res;
}

int main()
{
    ml::DataSet data = readMNISTData("/home/daiver/Downloads/train-images-idx3-ubyte");
    std::vector<int> labels = readMNISTLabels("/home/daiver/Downloads/train-labels-idx1-ubyte");
    ml::DataSet dataT = readMNISTData("/home/daiver/Downloads/t10k-images-idx3-ubyte");
    std::vector<int> labelsT = readMNISTLabels("/home/daiver/Downloads/t10k-labels-idx1-ubyte");

    srand(42);

    int countOfClasses = 10;
    int countOfFerns = 1400;
    int countOfFeaturesPerFern = 15;
    std::vector<Fern> ferns(countOfFerns);
    printf("Count of ferns %d Count of features %d\n", countOfFerns, countOfFeaturesPerFern);
    int memsizeOfProbsTables = countOfFerns * countOfClasses * pow(2, countOfFeaturesPerFern);
    printf("Count of probs %d mem %f Mb\n", memsizeOfProbsTables, memsizeOfProbsTables * 4.0 / 1024.0 / 1024.0);
    
    
    for(int i = 0; i < ferns.size(); ++i)
        trainFern(data, 28, 28, labels, countOfClasses, countOfFeaturesPerFern, ferns[i]);

    printf("Testing....\n");

    int countOfErrors = 0;
    for(int i = 0; i < labelsT.size(); ++i){
        int res = ml::bestFreq(getProbs(*dataT[i], 28, ferns, countOfClasses));
        int ans = labelsT[i];
        if(res != ans)
            countOfErrors++;
    }

    printf("Errors %d/%zu %f\n",
            countOfErrors, labelsT.size(), (float)countOfErrors/labelsT.size());
    printf("End\n");
    return 0;
}

auto func(const auto &a, const auto &b)
{
    return a + b * a + b;
}
