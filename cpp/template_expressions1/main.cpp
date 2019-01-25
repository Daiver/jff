#include <QVector>
#include <QDebug>

template<typename E, typename _Scalar>
class VecExpr
{
public:
    using Scalar = _Scalar;

    int nRows() const { return static_cast<E const &>(*this).nRows(); }
    Scalar operator()(const int row) const
    {
        return static_cast<E const &>(*this)(row);
    }
};

template<typename E1, typename E2, typename Op>
class VecBinaryOp : public VecExpr<VecBinaryOp<E1, E2, Op>, typename E1::Scalar>
{
public:
    using Scalar = typename E1::Scalar;
    VecBinaryOp(const E1 &e1, const E2 &e2, const Op &op): e1(e1), e2(e2), op(op)
    {
        Q_ASSERT(e1.nRows() == e2.nRows());
    }

    int nRows() const { return e1.nRows(); }
    Scalar operator()(const int row) const
    {
        return op(e1(row), e2(row));
    }

private:
    const E1 &e1;
    const E2 &e2;
    const Op &op;
};

template<typename E1, typename E2, typename Op>
VecBinaryOp<E1, E2, Op> vecBinaryOp(const E1 &e1, const E2 &e2, const Op &op)
{
    return VecBinaryOp<E1, E2, Op>(e1, e2, op);
}

template<typename _Scalar>
class Vector : public VecExpr<Vector<_Scalar>, _Scalar>
{
public:
    using Scalar = _Scalar;

    Vector() = default;
    Vector(const QVector<Scalar> &data): pData(data) {}

    template<typename E>
    Vector(const VecExpr<E, Scalar> &expr): pData(expr.nRows())
    {
        for(int i = 0; i < expr.nRows(); ++i){
            Scalar tmp = expr(i);
            (*this)(i) = tmp;
        }
    }

    QVector<Scalar> data() const { return pData; }

    int nRows() const
    {
        return pData.size();
    }

    Scalar &operator()(const int row)
    {
        return pData[row];
    }

    Scalar operator()(const int row) const
    {
        return pData[row];
    }

protected:
    QVector<Scalar> pData;
};


template<typename Scalar>
Scalar add(const Scalar a, const Scalar b)
{    
    return a + b;
}

template<typename Scalar>
Scalar sub(const Scalar a, const Scalar b)
{    
    return a - b;
}


template<typename Scalar>
Scalar mul(const Scalar a, const Scalar b)
{
    return a * b;
}

template<typename Scalar>
Scalar div(const Scalar a, const Scalar b)
{
    return a / b;
}


template<typename E1, typename E2>
auto operator +(const E1 &e1, const E2 &e2) -> decltype(vecBinaryOp(e1, e2, add<typename E1::Scalar>))
{
    return vecBinaryOp(e1, e2, add<typename E1::Scalar>);
}

template<typename Scalar>
auto operator +(
        const VecBinaryOp<Vector<Scalar>, Vector<Scalar>, mul<Scalar>> &e1,
        const VecBinaryOp<Vector<Scalar>, Vector<Scalar>, mul<Scalar>> &e2) -> decltype(vecBinaryOp(e1, e2, add<Scalar>))
{
//    return vecBinaryOp(e1, e2, add<typename E1::Scalar>);
}

template<typename E1, typename E2>
auto operator -(const E1 &e1, const E2 &e2) -> decltype(vecBinaryOp(e1, e2, sub<typename E1::Scalar>))
{
    return vecBinaryOp(e1, e2, sub<typename E1::Scalar>);
}

template<typename E1, typename E2>
auto operator *(const E1 &e1, const E2 &e2) -> decltype(vecBinaryOp(e1, e2, mul<typename E1::Scalar>))
{
    return vecBinaryOp(e1, e2, mul<typename E1::Scalar>);
}

template<typename E1, typename E2>
auto operator /(const E1 &e1, const E2 &e2) -> decltype(vecBinaryOp(e1, e2, div<typename E1::Scalar>))
{
    return vecBinaryOp(e1, e2, div<typename E1::Scalar>);
}

using Vectorf = Vector<float>;

int main(int , char *[])
{    
    Vectorf vec1({4});
    Vectorf vec2({5});
    Vectorf vec3({2});

    Vectorf res = vec1 + vec2 * vec3 / vec1;

    qDebug() << res.data();
    return 0;
}
