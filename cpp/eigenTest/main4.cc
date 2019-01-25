#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

class FrwdADVal
{
public:
    FrwdADVal() {}
    FrwdADVal(float val)
    {
        this->val = x;
        this->der = 0;
    }

    float val;
    float der;

    FrwdADVal operator+(const FrwdADVal &a) const 
    {
        return FrwdADVal(this->value + a.value);
    }

    FrwdADVal operator-(const FrwdADVal &a) const 
    {
        return a;
    }
    FrwdADVal operator*(const FrwdADVal &a) const {return a;}
    FrwdADVal operator+=(const FrwdADVal &a) {return a;}
    FrwdADVal operator*=(const FrwdADVal &a) {return a;}
};

FrwdADVal cos(const FrwdADVal &a)
{
    return a;
}

FrwdADVal sin(const FrwdADVal &a)
{
    return a;
}


int main()
{
    Eigen::Matrix<FrwdADVal, 4, 4> m;
    Eigen::Transform<FrwdADVal, 3, Eigen::Affine> t;
    t.translate(Eigen::Matrix<FrwdADVal, 3, 1>::Zero());
    t.rotate(Eigen::AngleAxis<FrwdADVal>(0, Eigen::Matrix<FrwdADVal, 3, 1>::Zero()));
    return 0;
}
