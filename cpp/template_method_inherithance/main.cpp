#include <stdio.h>
#include <iostream>
#include <string>

class Base
{
public:
    virtual void execute()
    {
        f<int>(10);
        f<std::string>("HELLO");
    }

    template<typename T>
    virtual void f(T x)
    {
        std::cout << "Base " << x << std::endl;
    }
};

class Derived1 : public Base
{
public:
    template<typename T>
    void f(T x)
    {
        std::cout << "Derived1 " << x << std::endl;
    }
};

int main()
{
    Base *b = new Base();
    b->execute();
    Derived1 d1;// = new Derived1();
    d1.execute();
    return 0;
}
