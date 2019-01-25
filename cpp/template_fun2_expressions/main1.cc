#include <stdio.h>
#include <iostream>
#include <vector>

// A CRTP base class for Vecs with a size and indexing.
// The template parameter is named 'E' for 'Expression'.
template <typename E>
class VecExpression {
  public:
    double operator[](size_t i) const { return static_cast<E const&>(*this)[i];     }
    size_t size()               const { return static_cast<E const&>(*this).size(); }

    // The following overload conversions to E, the template argument type;
    // e.g., for VecExpression<VecSum>, this is a conversion to VecSum.
    operator E&()             { return static_cast<      E&>(*this); }
    operator E const&() const { return static_cast<const E&>(*this); }
};

// The actual Vec class.
class Vec : public VecExpression<Vec> {
    std::vector<double> elems;

  public:
    double operator[](size_t i) const { return elems[i]; }
    double &operator[](size_t i)      { return elems[i]; }
    size_t size() const               { return elems.size(); }

    Vec(size_t n) : elems(n) {}

    // A Vec can be constructed from any VecExpression, forcing its evaluation.
    template <typename E>
    Vec(VecExpression<E> const& vec) : elems(vec.size()) {
        for (size_t i = 0; i != vec.size(); ++i) {
            elems[i] = vec[i];
        }
    }
};

template <typename E1, typename E2>
class VecSum : public VecExpression<VecSum<E1, E2> > {
    E1 const& _u;
    E2 const& _v;

public:
    VecSum(VecExpression<E1> const& u, VecExpression<E2> const& v) : _u(u), _v(v) {
        assert(u.size() == v.size());
    }

    double operator[](size_t i) const { return _u[i] + _v[i]; }
    size_t size()               const { return _v.size(); }
};
  
// Overloaded operator implementations are now templated on vector expression types.
template <typename E1, typename E2>
VecSum<E1,E2> const
operator+(VecExpression<E1> const& u, VecExpression<E2> const& v) {
   return VecSum<E1, E2>(u, v);
}

int main()
{
    return 0;
}
