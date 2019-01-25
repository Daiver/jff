#include <memory>
#include <QVariant>
#include <QDebug>

class MyClass
{
public:
    long x = 0l;

    MyClass()
    {
        qDebug() << "MyClass was created" << x;
    }

    ~MyClass()
    {
        qDebug() << "MyClass was destroyed" << x;
    }
};

int main(int, char *[])
{    
    MyClass *mc = new MyClass();

    std::shared_ptr<MyClass> ptr(mc);
    MyClass &value = *ptr;

    qDebug() << "before reutrn";
    return 0;
}
