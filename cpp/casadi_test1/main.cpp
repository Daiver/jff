#include <iostream>
#include <casadi/casadi.hpp>

using namespace casadi;

void test01()
{
    SX x = SX::sym("x");
    SX y = SX::sym("x", 5);
    SX z = SX::sym("x", 2, 5);
    SX f = mtimes(z, y);
    std::cout << x << std::endl;
    std::cout << f << std::endl;
    std::cout << f.size() << std::endl;
    std::cout << z(0, 1) << std::endl;
}

void test02()
{
    MX x = MX::sym("x");
    MX y = MX::sym("x", 5);
    MX z = MX::sym("x", 2, 5);
    MX f = mtimes(z, y);
    std::cout << x << std::endl;
    std::cout << f << std::endl;
    std::cout << f.size() << std::endl;
    std::cout << z(1, 3) << std::endl;
}

void test03()
{
    SX x = SX::sym("x");
    SX y = SX::sym("y");
    std::cout << x << y << std::endl;
    auto expr = x * x;
    auto grad = gradient(expr, x);
    Function f = Function("f", {x}, {expr});
    Function g = Function("g", {x}, {grad});

    //x.setInput(10);
    std::cout << f << std::endl;
    std::cout << f(DM(10)) << std::endl;
    std::cout << g << std::endl;
    std::cout << g(DM(10)) << std::endl;
}

int main()
{
    //test01();
    //std::cout << "=====================" << std::endl;
    //test02();
    test03();
    return 0;
}
