#include <iostream>

class Foo {
    public:
        int x;
        virtual void print();
};

class Bar: public Foo {
    public:
        int y;
        virtual void print();
};

void Foo::print() {
    std::cout << "x: " << this->x << '\n';
}

void Bar::print() {
    std::cout << "x: " << this->x << " y: " << this->y << '\n';
}

void print(Foo foo) {
    foo.print();
}

void print2(Foo &foo) {
    foo.print();
}

void print3(Foo *foo) {
    foo->print();
}

int main() {
    Bar bar;
    bar.x = 5;
    bar.y = 10;

    print(bar);
    print2(bar);
    print3(&bar);
}
