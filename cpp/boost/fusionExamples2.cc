#include <boost/fusion/tuple.hpp>
#include <boost/fusion/view.hpp>
#include <boost/fusion/algorithm.hpp>
#include <boost/fusion/adapted/struct/adapt_struct.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/zip.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/type_traits.hpp>
#include <boost/mpl/arg.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/proto/proto.hpp>
#include <string>
#include <iostream>

//using mpl = boost::mpl;
using namespace boost::lambda;
using namespace boost::fusion;
using namespace std;


class Student {
public:
    std::string name;
    int age;
    double averageMark;
    long weight;
};

BOOST_FUSION_ADAPT_STRUCT(
        Student,
        (std::string, name)
        (int, age)
        (double, averageMark)
        (long, weight)
    );

int main()
{
    Student s{"Vasya", 34, 2.3, 40};
    filter_view<Student, boost::is_integral<boost::mpl::arg<1>>> v{s};
    for_each(v, cout << _1 << '\n');

}
