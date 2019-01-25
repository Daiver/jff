#ifndef PARTICLEFILTER_H
#define PARTICLEFILTER_H

#include <stdlib.h>
#include <eigen3/Eigen/Dense>
#include <QDebug>
#include <QVector>
#include <QPair>

namespace ParticleFilter {

void mutate(QVector<Eigen::VectorXf> &samples, const QVector<QPair<float, float>> &intervals, float maxStep);

template<class CostFunction>
Eigen::VectorXf processParticleFilter(
        CostFunction costFunction, const QVector<QPair<float, float>> &intervals, int nPopulation, int nIter, float mutStep);

}




inline void ParticleFilter::mutate(QVector<Eigen::VectorXf> &samples, const QVector<QPair<float, float>> &intervals, float maxStep)
{
    for(int sampleInd = 0; sampleInd < samples.size(); ++sampleInd){
        Eigen::VectorXf &tmp = samples[sampleInd];
        for(int i = 0; i < tmp.rows(); ++i){
            tmp[i] += -maxStep/2.0 + maxStep * ((float)rand())/RAND_MAX;
            if(tmp[i] > intervals[i].second)
                tmp[i] = intervals[i].second;
            if(tmp[i] < intervals[i].first)
                tmp[i] = intervals[i].first;
        }
    }
}

template<class CostFunction>
Eigen::VectorXf ParticleFilter::processParticleFilter(
        CostFunction costFunction,
        const QVector<QPair<float, float>> &intervals,
        int nPopulation,
        int nIter,
        float mutStep)
{
    QVector<Eigen::VectorXf> population(nPopulation);
    for(int sampleInd = 0; sampleInd < population.size(); ++sampleInd){
        Eigen::VectorXf sample = Eigen::VectorXf::Zero(intervals.size());
        for(int i = 0; i < sample.rows(); ++i){
            float delta = intervals[i].second - intervals[i].first;
            sample[i] = intervals[i].first + delta * ((float)rand() / RAND_MAX);
        }
        population[sampleInd] = sample;
    }

    QVector<float> weights(nPopulation, 0);
    QVector<Eigen::VectorXf> newPopulation(nPopulation);
    for(int iterInd = 0; iterInd < nIter; ++iterInd){
        mutate(population, intervals, mutStep);

        float maxWeight = 0;
        for(int i = 0; i < population.size(); ++i){
            weights[i] = 1e6 - costFunction(population[i]);
            if(weights[i] > maxWeight)
                maxWeight = weights[i];

        }

        float beta = 0;
        int index = rand() % population.size();
        int newPopInd = 0;
        for(int sampleInd = 0; sampleInd < population.size(); ++sampleInd){
            beta += ((float)rand()/RAND_MAX) * 2.0 * maxWeight;
            while(beta > weights[index]){
                beta -= weights[index];
                index = (index + 1) % population.size();
            }
            newPopulation[newPopInd] = population[index];
            ++newPopInd;
        }
        population = newPopulation;
    }

    float maxWeight = -1e6;
    int maxWeightInd = 1;
    for(int i = 0; i < population.size(); ++i){
        float weight = -costFunction(population[i]);
        if(weight > maxWeight){
            maxWeight = weight;
            maxWeightInd = i;
        }
    }
    qDebug() << "Max Weights" << maxWeight;
    return population[maxWeightInd];
}




#endif // PARTICLEFILTER_H
