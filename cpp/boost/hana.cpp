#include <stdio.h>
#include <boost/hana.hpp>

namespace hana = boost::hana;

#include <cassert>
#include <iostream>
#include <string>
struct Fish { std::string name; };
struct Cat  { std::string name; };
struct Dog  { std::string name; };

int main()
{
    using namespace hana::literals;
    auto animals = hana::make_tuple(Fish{"Nemo"}, Cat{"Garfield"}, Dog{"Snoopy"});
    Cat garfield = animals[1_c];
    // Perform high level algorithms on tuples (this is like std::transform)
    auto names = hana::transform(animals, [](auto a) {
       return a.name;
    });
    assert(hana::reverse(names) == hana::make_tuple("Snoopy", "Garfield", "Nemo"));

    return 0;
}
