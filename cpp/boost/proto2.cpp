#include <iostream>
#include <boost/proto/proto.hpp>
#include <boost/typeof/std/ostream.hpp>
#include <cmath>
using namespace boost;


int main()
{
    // "sin" is a Proto terminal containing a function pointer
    proto::terminal< double(*)(double) >::type const sin = {&std::sin};
    double pi = 3.1415926535;
    proto::default_context ctx;
    // Create a lazy "sin" invocation and immediately evaluate it
    std::cout << proto::eval( 1 + 2 + sin(pi/2), ctx ) << std::endl;

    return 0;
}
