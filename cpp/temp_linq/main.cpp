#include <iostream>
#include <functional>
#include <vector>

template <typename Expr, typename ItemType>
class LinqExpression 
{
public:
    typedef ItemType ValueType;
    ItemType operator[](size_t i) const { return static_cast<Expr const&>(*this)[i];     }
    size_t size()                 const { return static_cast<Expr const&>(*this).size(); }

          Expr& operator()()       { return static_cast<      Expr&>(*this); }
    const Expr& operator()() const { return static_cast<const Expr&>(*this); }
};

template <typename Func, typename Expr, typename ItemType, typename InItemType>
class MapExpression : public LinqExpression <MapExpression<Func, Expr, ItemType, InItemType>, ItemType>
{
public:
    MapExpression(const Func &func, const Expr &expr): func(func), expr(expr){}

    ItemType operator[](size_t i) const { return func(expr[i]);}

private:
    const Func &func;
    const Expr &expr;
};

template <typename ItemType>
class VectorHolder : public LinqExpression <VectorHolder<ItemType>, ItemType>
{
public:
    VectorHolder(std::vector<ItemType> &vec): vec(vec) {}

    ItemType operator[](size_t i) const { return vec[i]; }
    size_t size() const { return vec.size(); }


private:
    std::vector<ItemType> &vec;
};

/*template<class T1, class T2, class Func, class Expr>
MapExpression<Func, Expr, T1, T2> map(Func &func, Expr const& expr)
{
    return MapExpression<Func, Expr, T1, T2>(func, expr);
}*/

template<class Func, class Expr>
auto map(const Func &func, Expr const& expr)
{
    typedef typename Expr::ValueType T2;
    typedef decltype(func(expr[0])) T1;
    //typedef decltype(Func::operator()(T2)) T1;
    return MapExpression<Func, Expr, T1, T2>(func, expr);
}

template<class Func>
class MapFunctor
{
public:
    MapFunctor(const Func &func): func(func){}

    template<typename Expr>
    auto operator()(Expr const& expr) const
    {
        typedef typename Expr::ValueType T2;
        typedef decltype(func(expr[0])) T1;
        return MapExpression<Func, Expr, T1, T2>(func, expr);
    }

private:
    const Func &func;
};

template<class Func, class Expr>
auto map(const Func &func)
{
    return MapFunctor<Func>(func);
}

template<class T>
VectorHolder<T> from(std::vector<T> &vec)
{
    return VectorHolder<T>(vec);
}

template<class Expr, class Func>
auto operator >>(const Expr &expr, Func& func)
{
    return func(expr);
}

int main()
{
    std::vector<int> vec = {1, 2, 3, 4, 5};
    auto square = [](int x) {return x*x;};
    //auto tmp = map<int, int>(square, from(vec));
    //auto tmp = map(square, from(vec));
    //auto tmp = map(square, from(vec));
    auto tmp = from(vec) >> map(square);
    //auto tmp = map<int, int>([](int x){return x*x;}, from(vec));
    std::cout << tmp[1] << std::endl;
    return 0;
}
