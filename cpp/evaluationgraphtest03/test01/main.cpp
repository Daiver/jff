#include <assert.h>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <typeindex>
#include <experimental/optional>

typedef float Scalar;
typedef std::map<std::string, Scalar> ValDict;

class Node;
typedef Node* NodePtr;
typedef std::vector<Node *> NodesPtr;

class AnalyticalDerivativeFunctor
{
public:
    virtual NodePtr operator()(const NodePtr &root) const = 0;
};

class Node
{
public:

    virtual const NodesPtr &args() const = 0;
    virtual Scalar evaluate(const ValDict &values) const = 0;
    virtual bool isConst() const = 0;

    virtual std::experimental::optional<const AnalyticalDerivativeFunctor *> analyticalDerivative() const { return {}; }

    bool isLeaf() const { return args().size(); }
};

class NodeWithoutArgs : public Node
{
public:
    const NodesPtr &args() const { return pArgs; }

private:
    NodesPtr pArgs;
};

class NumConst : public NodeWithoutArgs
{
public:
    NumConst(const Scalar value): pVal(value) {}

    Scalar evaluate(const ValDict &) const { return pVal; }
    bool isConst() const { return true; }

private:
    Scalar pVal;
};

class Value : public NodeWithoutArgs
{
public:
    Value(const std::string &name, const bool isVariable = false): pValName(name), pIsVar(isVariable) {}
    Scalar evaluate(const ValDict &values) const
    {
        return values.at(pValName);
    }

    bool isConst() const { return !this->pIsVar; }

private:
    std::string pValName;
    bool pIsVar;
};

class Operator : public Node {};

class OperatorWithManyArgs : public Operator
{
public:
    OperatorWithManyArgs(const NodesPtr &args): pArgs(args) {}
    const NodesPtr &args() const { return pArgs; }

private:
    NodesPtr pArgs;
};

bool isAllArgsConst(const NodesPtr &args)
{
    for(size_t i = 0; i < args.size(); ++i)
        if(!args[i]->isConst())
            return false;
    return true;
}

class OpAddAnalyticalDerivativeFunctor : public AnalyticalDerivativeFunctor
{
public:

};

class OpAdd : public OperatorWithManyArgs
{
public:
    OpAdd(const NodesPtr &args):  OperatorWithManyArgs(args) {}
    Scalar evaluate(const ValDict &values) const
    {
        const int nArgs = this->args().size();
        assert(nArgs > 0);
        Scalar res = this->args()[0]->evaluate(values);
        for(int i = 1; i < nArgs; ++i)
            res += this->args()[i]->evaluate(values);
        return res;
    }
    bool isConst() const { return isAllArgsConst(this->args()); }

    std::experimental::optional<const AnalyticalDerivativeFunctor *> analyticalDerivative() const
    {

    }
};

class OpMul : public OperatorWithManyArgs
{
public:
    OpMul(const NodesPtr &args):  OperatorWithManyArgs(args) {}
    Scalar evaluate(const ValDict &values) const
    {
        const int nArgs = this->args().size();
        assert(nArgs > 0);
        Scalar res = this->args()[0]->evaluate(values);
        for(int i = 1; i < nArgs; ++i)
            res *= this->args()[i]->evaluate(values);
        return res;
    }
    bool isConst() const { return isAllArgsConst(this->args()); }
};

int main()
{
    const auto a2 = new NumConst(2);
    const auto x = new Value("x");

    const auto graph = OpAdd({a2, x});

    return 0;
}
