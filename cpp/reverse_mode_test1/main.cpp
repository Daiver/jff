#include <QtGlobal>
#include <iostream>
#include <memory>

#include "Eigen/Core"

using Scalar = float;
using Matrix = Eigen::Matrix<float, -1, -1>;
using ConstMatrixMap = const Eigen::Map<const Matrix>;
using ConstMatrixRef = const Matrix &;
using ConstSensitivity = const Eigen::Map<const Eigen::VectorXf>;
using ConstSensitivityRef = const Eigen::Map<const Eigen::VectorXf> &;

class Node;

using NodePtr = std::shared_ptr<Node>;

class Node
{
public:
    Node(const bool requiresGrad = false)
    {
        this->m_requiresGrad = requiresGrad;
    }

    Node(const Node &other) = default;

    virtual ~Node() {}

    virtual ConstMatrixRef value() const = 0;
    virtual ConstMatrixRef grad() const { Q_ASSERT(false);/*Not implemented*/}
    void backward(const Scalar sensetivity)
    {
        Eigen::VectorXf sensVec(1);
        sensVec[0] = sensetivity;
        ConstSensitivity sensMap(sensVec.data(), 1, 1);
        this->backward(sensMap);
    }
    virtual void backward(ConstSensitivityRef sensetivity) = 0;
    virtual void zeroGrad() {}

    bool requiresGrad() const { return m_requiresGrad; }
    void setRequiresGrad(bool requiresGrad) { m_requiresGrad = requiresGrad; }

protected:
    bool m_requiresGrad;
};

class VarNode : public Node
{
public:
    VarNode(): Node(false)
    {
        this->m_value.setConstant(0);
        this->zeroGrad();
    }

    VarNode(const Scalar value, const bool requiresGrad = false):
        Node(requiresGrad)
    {
        this->m_value.resize(1, 1);
        this->m_value(0, 0) = value;
        this->m_grad.resize(1, 1);
        this->zeroGrad();
    }

    VarNode(ConstMatrixRef value, const bool requiresGrad = false):
        Node(requiresGrad)
    {
        this->m_value = value;
        this->m_grad.setZero(this->m_value.rows(), this->m_value.cols());
        this->zeroGrad();
    }

    virtual ConstMatrixRef value() const {return m_value;}
    virtual void backward(ConstSensitivityRef sensetivity)
    {
        const Eigen::Map<const Eigen::MatrixXf> mapSense(sensetivity.data(), this->m_grad.rows(), this->m_grad.cols());
        this->m_grad += mapSense;
    }
    virtual void zeroGrad() { this->m_grad.setConstant(0); }

    ConstMatrixRef grad() const { return m_grad; }

protected:
    Matrix m_value;
    Matrix m_grad;
};

class ValuableNode : public Node
{
public:
    ValuableNode(const bool requiresGrad): Node(requiresGrad) {}
    ConstMatrixRef value() const {return m_value;}
protected:
    Matrix m_value;
};

class BinNode : public ValuableNode
{
public:
    BinNode(const NodePtr &lhs, const NodePtr &rhs):
        ValuableNode(lhs->requiresGrad() || rhs->requiresGrad())
    {
        m_lhs = lhs;
        m_rhs = rhs;
    }

protected:
    NodePtr m_lhs;
    NodePtr m_rhs;
};

class AddNode : public BinNode
{
public:
    AddNode(const NodePtr &lhs, const NodePtr &rhs):
        BinNode(lhs, rhs)
    {
        this->m_value = lhs->value() + rhs->value();
    }

    virtual void backward(ConstSensitivityRef sensetivity)
    {
        this->m_lhs->backward(sensetivity);
        this->m_rhs->backward(sensetivity);
    }
};

class MulNode : public BinNode
{
public:
    MulNode(const NodePtr &lhs, const NodePtr &rhs):
        BinNode(lhs, rhs)
    {
        this->m_value = lhs->value().array() * rhs->value().array();
    }

    virtual void backward(ConstSensitivityRef sensetivity)
    {
        ConstMatrixMap senseMap(sensetivity.data(), m_lhs->value().rows(), m_lhs->value().cols());
        const Matrix lhsSense = senseMap.array() * m_rhs->value().array();
        const Matrix rhsSense = senseMap.array() * m_lhs->value().array();
        ConstSensitivity lhsSenseMap(lhsSense.data(), lhsSense.nonZeros(), 1);
        ConstSensitivity rhsSenseMap(rhsSense.data(), rhsSense.nonZeros(), 1);
        this->m_lhs->backward(lhsSenseMap);
        this->m_rhs->backward(rhsSenseMap);
    }
};

class RowNode : public ValuableNode
{
public:
    RowNode(const NodePtr &node, const long rowInd): ValuableNode(node->requiresGrad())
    {
        Q_ASSERT(false);//Not implemented
        this->m_value = node->value().row(rowInd);
    }

    virtual void backward(ConstSensitivityRef sensetivity)
    {
        ConstMatrixMap senseMap(sensetivity.data(), this->value().rows(), this->value().cols());
//        Eigen::MatrixXf newSense =
    }
};

class Variable
{
public:
    Variable() = default;

    Variable(const Scalar value, const bool requiresGrad = false)
    {
        this->m_node = std::make_shared<VarNode>(VarNode(value, requiresGrad));
    }

    Variable(ConstMatrixRef value, const bool requiresGrad = false)
    {
        this->m_node = std::make_shared<VarNode>(VarNode(value, requiresGrad));
    }

    Variable(const NodePtr &node)
    {
        this->m_node = node;
    }

    template<typename NodeType>
    Variable(const NodeType &node)
    {
        this->m_node = std::make_shared<NodeType>(node);
    }

    NodePtr node() const { return m_node; }

    ConstMatrixRef value() const { return m_node->value(); }
    ConstMatrixRef grad() const { return m_node->grad(); }
    void backward(ConstSensitivityRef sensetivity) { m_node->backward(sensetivity); }
    void backward(const Scalar sensetivity) { m_node->backward(sensetivity); }
    void zeroGrad() { m_node->zeroGrad(); }

protected:
    NodePtr m_node;
};

Variable operator +(const Variable &lhs, const Variable &rhs)
{
    return Variable(AddNode(lhs.node(), rhs.node()));
}

Variable operator *(const Variable &lhs, const Variable &rhs)
{
    return Variable(MulNode(lhs.node(), rhs.node()));
}


int main()
{
    Eigen::MatrixXf xVal(2, 3);
    xVal << 1, 2, 3, 4, 5, 6;

    Eigen::MatrixXf yVal(2, 3);
    yVal << 0, 0, 0, 0, 0, 0;

    Eigen::MatrixXf zVal(2, 3);
    zVal << 1, 1, 1, 1, 1, 1;

    Variable x(xVal, true);
    Variable y(yVal, true);
    Variable z(zVal, true);

    auto res = (x + y) * z;
    res.backward(1);
    std::cout << "res " << res.value() << std::endl;
    std::cout << "x.grad " << x.grad() << std::endl;
    std::cout << "y.grad " << y.grad() << std::endl;
    std::cout << "z.grad " << z.grad() << std::endl;

    return 0;
}
