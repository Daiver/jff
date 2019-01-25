#include <stdio.h>
#include <string>
#include <fstream>
#include <algorithm>

#include "common.h"
#include "ferns.h"

void readCSVIntoVector(const std::string &fname, std::vector<std::vector<float>> &data)
{
    std::ifstream in(fname.c_str());
    //in.open();
    std::string line;
    std::getline(in, line);
    //printf("%s\n", line.c_str());
    while(!in.eof()){
        std::getline(in, line);
        //printf("%s\n", line.c_str());
        std::vector<std::string> tmp = split(line, ';');
        if(tmp.size() == 0)
            continue;
        data.push_back(std::vector<float>(tmp.size()));
        for(int i = 0; i < tmp.size(); ++i)
            data[data.size() - 1][i] = atof(tmp[i].c_str());
    }
    in.close();
}

void readCSVCommaIntoVector(const std::string &fname, std::vector<std::vector<float>> &data)
{
    std::ifstream in(fname.c_str());
    //in.open();
    std::string line;
    std::getline(in, line);
    //printf("%s\n", line.c_str());
    while(!in.eof()){
        std::getline(in, line);
        //printf("%s\n", line.c_str());
        std::vector<std::string> tmp = split(line, ',');
        if(tmp.size() == 0)
            continue;
        data.push_back(std::vector<float>(tmp.size()));
        for(int i = 0; i < tmp.size(); ++i)
            data[data.size() - 1][i] = atof(tmp[i].c_str());
    }
    in.close();
}


void linearTest01()
{
    DataSet ds = {
        {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {7.5}, {8}, {9}, {10}
    };
    std::vector<float> values = {
         0,   2,    4,   6,   8,   10,  12,  14,  15,  16,  18, 20
    };
    std::vector<Fern> cascade;
    trainCascade(ds, values, 1, 500, 55, cascade);
    DataSet test = {
        {0}, {0.5}, {1.5}, {7.5}, {5.5}, {8}, {3.5}, {9.5}, {6.5}
    };

/*    for(auto &&x : cascade)*/
        /*x.print();*/

    for(auto &&x : test)
        printf("x:%f %f\n", x[0], activateCascade(cascade, x));
}

void winTest01()
{
    std::vector<std::vector<float>> rawData;
    //readCSVIntoVector("./winequality-red.csv", rawData);
    readCSVIntoVector("./winequality-white.csv", rawData);
/*    for(int i = 0; i < rawData.size(); ++i){*/
        //for(int j = 0; j < rawData[i].size(); ++j)
            //printf("%f ", rawData[i][j]);
        //printf("\n");
    /*}*/

    shuffle(&rawData, 10000);
    DataSet trainDs, testDs;
    std::vector<float> trainVals, testVals;

    int testSize = 1000;
    for(int i = 0; i < testSize; ++i){
        testDs.push_back(std::vector<float>(rawData[i].size() - 1, 0));
        for(int j = 0; j < rawData[i].size() - 1; ++j)
            testDs[testDs.size() - 1][j] = rawData[i][j];
        testVals.push_back(rawData[i][rawData[i].size() - 1]);
    }
    for(int i = testSize; i < rawData.size(); ++i){
        trainDs.push_back(std::vector<float>(rawData[0].size() - 1, 0));
        for(int j = 0; j < rawData[i].size() - 1; ++j)
            trainDs[trainDs.size() - 1][j] = rawData[i][j];
        trainVals.push_back(rawData[i][rawData[i].size() - 1]);
    }
    printf("End reading\n");

    std::vector<Fern> cascade;
    trainCascade(trainDs, trainVals, 11, 84, 1048, cascade);
    float err = 0;
    int intErr = 0;
    for(int i = 0; i < testDs.size(); ++i){
        float res = activateCascade(cascade, testDs[i]);
        float ans = testVals[i];
        err += fabs(res - ans);
        if(fabs(res - ans) > 0.5)
            intErr++;
        //printf("%f %f\n", res, testVals[i]);
    }
    printf("Err %f %d/%d\n", err/testVals.size(), intErr, testSize);
   
/*    printVecHor(testVals);*/
    /*printVec2D(testDs);*/
}

void blogTest01()
{
    std::vector<std::vector<float>> rawData;
    //readCSVIntoVector("./winequality-red.csv", rawData);
    readCSVCommaIntoVector("/home/daiver/Downloads/BlogData/all.csv", rawData);
/*    for(int i = 0; i < rawData.size(); ++i){*/
        //for(int j = 0; j < rawData[i].size(); ++j)
            //printf("%f ", rawData[i][j]);
        //printf("\n");
    /*}*/

    shuffle(&rawData, 80000);
    DataSet trainDs, testDs;
    std::vector<float> trainVals, testVals;

    int testSize = 5000;
    for(int i = 0; i < testSize; ++i){
        testDs.push_back(std::vector<float>(rawData[i].size() - 1, 0));
        for(int j = 0; j < rawData[i].size() - 1; ++j)
            testDs[testDs.size() - 1][j] = rawData[i][j];
        testVals.push_back(rawData[i][rawData[i].size() - 1]);
    }
    for(int i = testSize; i < rawData.size(); ++i){
        trainDs.push_back(std::vector<float>(rawData[0].size() - 1, 0));
        for(int j = 0; j < rawData[i].size() - 1; ++j)
            trainDs[trainDs.size() - 1][j] = rawData[i][j];
        trainVals.push_back(rawData[i][rawData[i].size() - 1]);
    }

    float mean = 0;
    for(int i = 0; i < rawData.size(); ++i)
        mean += rawData[i][rawData[0].size() - 1];
    mean /= rawData.size();
    float std = 0;
    for(int i = 0; i < rawData.size(); ++i)
        std += fabs(mean - rawData[i][rawData[0].size() - 1]);
    std /= rawData.size();

    rawData.clear();
    printf("End reading\n");

    std::vector<Fern> cascade;
    //trainCascade(trainDs, trainVals, 10, 500, 10, cascade);
    float err = 0;
    int intErr = 0;
    std::vector<float> errors(testVals.size(), 0);
    for(int i = 0; i < testDs.size(); ++i){
        float res = activateCascade(cascade, testDs[i]);
        float ans = testVals[i];
        err += fabs(res - ans);
        errors[i] = fabs(res - ans);
        if(fabs(res - ans) > 0.5)
            intErr++;
        //printf("%f %f\n", res, testVals[i]);
    }

    std::sort(errors.begin(), errors.end());
/*    for(int i = 0; i < errors.size(); ++i)*/
        /*printf("%d %f %f\n", i, errors[i], testVals[i]);*/

    printf("Median eror %f\n", errors[errors.size()/2]);
    printf("Err %f %d/%d\n", err/testVals.size(), intErr, testSize);
    printf("Mean %f Std %f\n", mean, std);
   
/*    printVecHor(testVals);*/
    /*printVec2D(testDs);*/
}
int main()
{
    srand(42);
    //linearTest01();
    winTest01();
    //blogTest01();
    return 0;
}
