#include <stdio.h>

template<class E>
class FrwdADExpression {
public:
    operator E&()             {return static_cast<      E&>(*this);}
    operator E const&() const {return static_cast<const E&>(*this);}

    double val() const {return static_cast<const E&>(*this).val();}
    double der() const {return static_cast<const E&>(*this).der();}
};

class FrwdADVal : public FrwdADExpression<FrwdADVal>
{
public:
    FrwdADVal(const double val, const double der):
        m_val(val), m_der(der) {}

    double val() const 
    {
        return m_val;
    }

    double der() const 
    {
        return m_der;
    }

    double m_val;
    double m_der;
};



int main()
{

}
