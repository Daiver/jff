#include <memory>
#include <QVariant>
#include <QDebug>

template<typename T>
class SomeMat
{
public:
    SomeMat()
    {
        qDebug() << "SomeMat was created!";
    }

    SomeMat(T x): x(x)
    {
        qDebug() << "SomeMat was created!" << x;
    }

    SomeMat(SomeMat && ) = default;
    SomeMat(SomeMat &) = default;
//    SomeMat(const SomeMat &) = default;

    SomeMat(const SomeMat &someMat)
    {
        this->x = someMat.x;
        qDebug() << "SomeMat was copied!" << x;
    }

    ~SomeMat()
    {
        qDebug() << "SomeMat was Deleted!" << x;
    }

    T x;
};

typedef SomeMat<int> SomeMati;
typedef SomeMat<long> SomeMatl;

typedef std::shared_ptr<SomeMati> SomeMatiPtr;
typedef std::shared_ptr<SomeMatl> SomeMatlPtr;

Q_DECLARE_METATYPE(SomeMatiPtr);
Q_DECLARE_METATYPE(SomeMatlPtr);

int main(int , char *[])
{    
    SomeMati matI(2);
    SomeMatl matL(3);
    qDebug() << "Before variant construction";
    QVariant variant = QVariant::fromValue(std::make_shared<SomeMati>(matI));
    qDebug() << "Before second variant construction";
    variant.setValue(std::make_shared<SomeMatl>(matL));

    SomeMatl &matLRef = *qvariant_cast<SomeMatlPtr>(variant);
    SomeMatl &matLRef2 = *qvariant_cast<SomeMatlPtr>(variant);
    qDebug() << "ref val" << matLRef.x;
    qDebug() << "ref2 val" << matLRef2.x;

    qDebug() << "Before return";
    return 0;
}
