#include <boost/bind.hpp>
#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include "boost/python/numeric.hpp"
#include "boost/python/extract.hpp"
#include <iostream>

#include <vector>

char const* greet()
{
   return "hello, world";
}

int greet2(int i)
{
    return i*10;
}

boost::python::list greet3(int i)
{
    auto l = boost::python::list();
    for(int j = 0; j < i; ++j)
        l.append(10);
    return l;
    //return new int[i];
}

void setArray(boost::python::numeric::array data) {
    // Access a built-in type (an array)
    boost::python::numeric::array a = data;
    // Need to <extract> array elements because their type is unknown
    std::cout << "First array item: " << boost::python::extract<int>(a[0]) << std::endl;
}

BOOST_PYTHON_MODULE(libpblib)
{
    using namespace boost::python;
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

    def("greet", greet);
    def("greet2", greet2);
    def("greet3", greet3);
    def("setArray", &setArray);
}
