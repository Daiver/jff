
#include <array>
#include <set>
#include <map>
#include <cassert>
#include <string>
#include <iostream>
#include <vector>
#include <typeinfo>


//Utils

template<typename ItemType, typename Container>
bool contains(const Container &container, const ItemType &item)
{
    return container.find(item) != container.end();
}

template<typename Container>
void extend(const Container &containerToAppend, Container &containerToExtend)
{
    containerToExtend.insert(std::end(containerToExtend), std::begin(containerToAppend), std::end(containerToAppend));
}

template<typename T1, typename Func>
auto map(const Func &func, const std::vector<T1> &vec)
{
    std::vector<decltype(func(vec[0]))> res(vec.size());
    for(size_t i = 0; i < vec.size(); ++i)
        res[i] = func(vec[i]);
    return res;
}

template<typename T1, typename Predicate>
bool all(const Predicate &predicate, const std::vector<T1> &vec)
{
    bool res = true;
    for(const auto &x : vec)
        res = res && predicate(x);
    return res;
}

template<typename T1, typename Predicate>
bool any(const Predicate &predicate, const std::vector<T1> &vec)
{
    bool res = false;
    for(const auto &x : vec)
        res = res || predicate(x);
    return res;
}

template<typename T1, typename Predicate>
bool justOne(const Predicate &predicate, const std::vector<T1> &vec)
{
    bool res = false;
    for(const auto &x : vec){
        const bool localRes = predicate(x);
        res = !res && localRes;
        if(!res && localRes)
            break;
    }
    return res;
}

//Curvature
enum class Curvature
{
    Constant,
    Linear,
    Convex,
    Concave,
    Unknown
};

const int N_CURVATURETYPES = 5;

bool atLeastsConst(const Curvature &curvature)
{
    return curvature == Curvature::Constant;
}

bool atLeastLinear(const Curvature &curvature)
{
    return curvature == Curvature::Linear || curvature == Curvature::Constant;
}

bool atLeastConvex(const Curvature &curvature)
{
    return curvature == Curvature::Convex || atLeastLinear(curvature);
}

bool atLeastConcave(const Curvature &curvature)
{
    return curvature == Curvature::Concave || atLeastLinear(curvature);
}

std::array<uint, N_CURVATURETYPES> curvatureFreqs(const std::vector<Curvature> &curvatures)
{
    std::array<uint, N_CURVATURETYPES> res;
    res.fill(0);
    for(const auto &x : curvatures)
        ++res[int(x)];

    return res;
}

Curvature findMinimumCommonCurvature(const std::vector<Curvature> &curvatures)
{
    assert(curvatures.size() > 0);
    const auto curvFreqs  = curvatureFreqs(curvatures);
    const bool isConstant = (curvFreqs[size_t(Curvature::Constant)]                                                                               ) == curvatures.size();
    const bool isLinear   = (curvFreqs[size_t(Curvature::Constant)] + curvFreqs[size_t(Curvature::Linear)]                                        ) == curvatures.size();
    const bool isConvex   = (curvFreqs[size_t(Curvature::Constant)] + curvFreqs[size_t(Curvature::Linear)] + curvFreqs[size_t(Curvature::Convex)] ) == curvatures.size();
    const bool isConcave  = (curvFreqs[size_t(Curvature::Constant)] + curvFreqs[size_t(Curvature::Linear)] + curvFreqs[size_t(Curvature::Concave)]) == curvatures.size();
    if(isConstant)
        return Curvature::Constant;
    if(isLinear)
        return Curvature::Linear;
    if(isConvex)
        return Curvature::Convex;
    if(isConcave)
        return Curvature::Concave;
    return Curvature::Unknown;
}

namespace std {
    std::string to_string(const Curvature &curv)
    {
        switch (curv) {
        case Curvature::Constant:
            return "Constant";
        case Curvature::Linear:
            return "Linear";
        case Curvature::Convex:
            return "Convex";
        case Curvature::Concave:
            return "Concave";
        case Curvature::Unknown:
            return "Unknown";
        }
        assert(false);//Should not reach this
        return "Unknown";
    }
}

//Node

class Node
{
public:
    virtual Curvature curvatureWrtVars(const std::set<std::string> &vars) const = 0;
    virtual Node *clone() const = 0;
    virtual std::string toString() const = 0;

    virtual bool isHomogeneous() const { return false; }

    const std::vector<Node *> &arguments() const { return pArguments; }
    void setArguments(const std::vector<Node *> &newArgs) { this->pArguments = newArgs; }
    bool isLeaf() const { return arguments().size() == 0; }
protected:
    std::vector<Node*> pArguments;
};

class Placeholder : public Node
{
public:
    Placeholder(const std::string &name): pName(name) {        }

    Curvature curvatureWrtVars(const std::set<std::__cxx11::string> &vars) const
    {
        if(contains(vars, this->name()))
            return Curvature::Linear;
        return Curvature::Constant;
    }

    Placeholder *clone() const
    {
        return new Placeholder(*this);
    }

    std::string toString() const
    {
        return this->name();
    }

    std::string name() const { return pName; }

private:
    std::string pName;
};

bool isNodeDependsOn(const std::set<std::__cxx11::string> &vars, const Node *node)
{
    const Placeholder *placeholder = dynamic_cast<const Placeholder *>(node);
    if(placeholder != NULL){
        return contains(vars, placeholder->name());
    }
    for(const auto &x : node->arguments())
        if(isNodeDependsOn(vars, x))
            return true;
    return false;
}

bool isNodeConst(const std::set<std::__cxx11::string> &vars, const Node *node)
{
    return !isNodeDependsOn(vars, node);
}

class OpAdd : public Node
{
public:
    OpAdd(const std::vector<Node *> &args)
    {
        assert(args.size() > 0);
        this->pArguments = args;
    }

    Curvature curvatureWrtVars(const std::set<std::__cxx11::string> &vars) const
    {
        const auto curvatures = map([&](const auto x){return x->curvatureWrtVars(vars);}, this->arguments());
        return findMinimumCommonCurvature(curvatures);
    }

    OpAdd *clone() const
    {
        return new OpAdd(*this);
    }

    std::string toString() const
    {
        auto res = arguments()[0]->toString();
        for(size_t i = 1; i < arguments().size(); ++i)
            res += " + " + arguments()[i]->toString();
        return "(" + res + ")";
    }

    virtual bool isHomogeneous() const { return true; }
};

class OpMul : public Node
{
public:
    OpMul(const std::vector<Node *> &args)
    {
        assert(args.size() > 0);
        this->pArguments = args;
    }

    Curvature curvatureWrtVars(const std::set<std::__cxx11::string> &vars) const
    {
        const auto curvatures = map([&](const auto x){return x->curvatureWrtVars(vars);}, this->arguments());
        const auto curvFreqs  = curvatureFreqs(curvatures);
        const size_t nConstantFreqs = curvFreqs[size_t(Curvature::Constant)];
        if(nConstantFreqs == curvatures.size())
            return Curvature::Constant;
        if(nConstantFreqs < curvatures.size() - 1)
            return Curvature::Unknown;
        if(curvFreqs[size_t(Curvature::Linear)] == 1)
            return Curvature::Linear;
        if(curvFreqs[size_t(Curvature::Convex)] == 1)
            return Curvature::Convex;
        if(curvFreqs[size_t(Curvature::Concave)] == 1)
            return Curvature::Concave;
        //should not reach this;
        assert(false);
        return Curvature::Unknown;
    }

    OpMul *clone() const
    {
        return new OpMul(*this);
    }

    std::string toString() const
    {
        auto res = arguments()[0]->toString();
        for(size_t i = 1; i < arguments().size(); ++i)
            res += " * " + arguments()[i]->toString();
        return "(" + res + ")";
    }

    virtual bool isHomogeneous() const { return true; }
};

std::vector<Node *> liftHomogeneousAssociativeOperation(const Node *root)
{
    std::vector<Node *> res;
    const auto args = root->arguments();
    for(Node *arg : args){
        const bool canBeMerged = (typeid(*arg) == typeid(*root)) && arg->isHomogeneous();
        if(!canBeMerged){
            res.push_back(arg);
            continue;
        }
        std::vector<Node *> liftedLocalAdds = liftHomogeneousAssociativeOperation(arg);
        res.insert(std::end(res), std::begin(liftedLocalAdds), std::end(liftedLocalAdds));
    }

    return res;
}

Node *liftHomogeneousAssociativeOperationOfGraph(const Node *root)
{
    if(root->isLeaf())
        return root->clone();
    std::vector<Node *> liftedArgs = liftHomogeneousAssociativeOperation(root);
    for(int i =0; i < liftedArgs.size(); ++i)
        liftedArgs[i] = liftHomogeneousAssociativeOperationOfGraph(liftedArgs[i]);
    Node *res = root->clone();
    res->setArguments(liftedArgs);
    return res;
}

std::vector<Node *> rearangeArgs(const std::set<std::__cxx11::string> &vars, const std::vector<Node *> &unorderedArguments, const bool constsLast = true)
{
    std::vector<Node *> constantArgs;
    std::map<std::string, std::vector<Node *>> argsByVar;
    for(const auto varName: vars){
        for(size_t aInd = 0; aInd < unorderedArguments.size(); ++aInd){
            Node *arg = unorderedArguments[unorderedArguments.size() - 1 - aInd];
            if(isNodeDependsOn({varName}, arg)){
                argsByVar[varName].push_back(arg);
            }
        }
    }

    std::vector<Node *> orderedArguments;
    if(!constsLast)
        extend(constantArgs, orderedArguments);
    for(const auto varName: vars)
        extend(orderedArguments, argsByVar[varName]);

    if(constsLast)
        extend(constantArgs, orderedArguments);
    return orderedArguments;
}

OpMul *rearangeMulArgs(const std::set<std::__cxx11::string> &vars, const OpMul *node)
{
    auto orderedArgs = rearangeArgs(vars, node->arguments(), false);
    return new OpMul(orderedArgs);
}

OpAdd *rearangeAddArgs(const std::set<std::__cxx11::string> &vars, const OpAdd *node)
{
    auto orderedArgs = rearangeArgs(vars, node->arguments(), true);
    return new OpAdd(orderedArgs);
}

Node *rearangeAndMergeMulAndAddArgsOnGraph(const std::set<std::string> &vars, const Node *root)
{
    std::vector<Node *> args = root->arguments();
    for(size_t i = 0; i < args.size(); ++i)
        args[i] = rearangeAndMergeMulAndAddArgsOnGraph(vars, args[i]);
    const OpAdd *opAdd = dynamic_cast<const OpAdd *>(root);
    const OpMul *opMul = dynamic_cast<const OpMul *>(root);
    std::vector<Node *> orderedArgs;
    if(opAdd != NULL){
        orderedArgs = rearangeArgs(vars, args, true);
    }else if(opMul != NULL){
        orderedArgs = rearangeArgs(vars, args, false);
    }

    Node *res = root->clone();
    res->setArguments(orderedArgs);;
    return res;
}

Node *mkGraphNormalLSForm(const Node *root)
{
    assert(false);
}

int main(int , char *[])
{
    Placeholder x("x");
    Placeholder y("y");
    Placeholder z("z");
    Placeholder a("a");
    Placeholder b("b");
    Placeholder c("c");
    OpMul *tmp = dynamic_cast<OpMul *>(&x);
//    auto graph = OpAdd({&x, &y});
//    auto graph = OpMul({new OpAdd({&x, &y}), new OpAdd({&y, &z})});
//    auto graph = new OpMul({new OpMul({&x, &y}), new OpAdd({&y, &z})});
//    auto graph2 = mergeMuls(graph);

    OpMul *graph = new OpMul({new OpMul({&a, &b}), new OpAdd({&y, &z})});
    auto graph2 = liftHomogeneousAssociativeOperationOfGraph(graph);
    auto graph3 = rearangeAndMergeMulAndAddArgsOnGraph({"x", "y"}, graph2);

    std::cout << graph->toString() << std::endl;
    std::cout << graph2->toString() << std::endl;
    std::cout << graph3->toString() << std::endl;
//    std::cout << std::to_string(graph.curvatureWrtVars({"x", "y"})) << std::endl;
//    std::cout << std::to_string(graph.curvatureWrtVars({"x"})) << std::endl;
//    std::cout << std::to_string(graph.curvatureWrtVars({"y"})) << std::endl;
//    std::cout << std::to_string(graph.curvatureWrtVars({"z"})) << std::endl;
//    std::cout << std::to_string(graph.curvatureWrtVars({})) << std::endl;
    return 0;
}
