#ifndef NODE_H
#define NODE_H

#include <iostream>
#include <Eigen/Core>

template<class Scalar, class ReturnValue>
class Node
{
public:
    virtual ReturnValue predict(const Eigen::Matrix<Scalar, -1, 1> &sample) = 0;
    virtual void print(const std::string &prefix = "") const = 0;
};

template<class Scalar, class ReturnValue>
class NodeBranch : public Node<Scalar, ReturnValue>
{
public:
    ReturnValue predict(
            const Eigen::Matrix<Scalar, -1, 1> &sample)
    {
        if(sample[featInd] >= threshold)
            return right->predict(sample);
        return left->predict(sample);
    };

    void print(const std::string &prefix) const 
    {
        std::cout << prefix << "Branch: "
                  << "featInd:" << featInd
                  << " thr:" << threshold
                  << std::endl;
        this->left ->print(prefix + "  ");
        this->right->print(prefix + "  ");
    }

    Node<Scalar, ReturnValue> *left;
    Node<Scalar, ReturnValue> *right;
    Scalar threshold;
    int featInd;
};

template<class Scalar, class ReturnValue>
class NodeLeaf : public Node<Scalar, ReturnValue>
{
public:
    ReturnValue predict(
            const Eigen::Matrix<Scalar, -1, 1> &sample)

    {
        return value;
    }

    void print(const std::string &prefix) const 
    {
        std::cout << prefix << "Leaf: value: " << this->value << std::endl;
    }

    ReturnValue value;
};

#endif
