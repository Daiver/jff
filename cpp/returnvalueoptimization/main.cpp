#include <iostream>

struct C {
int i;
  C() {std::cout << "constr\n";}
  C(C&) { std::cout << "A copy was made.\n"; }
//private:
  C(const C&) { std::cout << "A copy was made.\n"; }
};

C f() {
    C tmp;
    tmp.i = 10;
    return tmp;
}

int main() {
  std::cout << "Hello World!\n";
  C obj = f();
  std::cout << (f().i + 5);
  //C obj2 = obj;
  return 0;
}
