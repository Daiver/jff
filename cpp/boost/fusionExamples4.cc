#include <iostream>
#include <string>

#include <boost/fusion/tuple.hpp>
#include <boost/fusion/view.hpp>
#include <boost/fusion/algorithm.hpp>
#include <boost/fusion/include/zip.hpp>
#include <boost/type_traits.hpp>
#include <boost/mpl/arg.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/fusion/adapted/struct/adapt_struct.hpp>


#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/mpl/range_c.hpp>

template<class Struct, class Index> 
std::string nameOfField()
{
    return boost::fusion::extension::struct_member_name<Struct, Index::value>::call();
}

template <class t>
struct print_pair {
    print_pair(const t &seq) : seq_(seq) {}

    template <typename i>
    void operator()(i) const
    {
        std::cout 
            << boost::fusion::extension::struct_member_name<t, i::value>::call() 
            << ": " 
            << boost::fusion::at<i>(seq_) << "\n";
    }

    const t &seq_;
};

template<class t>
void print_struct(const t &value)
{
    typedef boost::mpl::range_c<int, 0, boost::fusion::result_of::size<t>::value> indices;
    print_pair<t> printer(value);
    boost::fusion::for_each(indices(), [value](auto i_){
        std::cout 
            << nameOfField<t, decltype(i_)>()
            << ": " 
            << boost::fusion::at<decltype(i_)>(value) << "\n";
            });
}


struct strct {
    int a;
    double b;
    char c;
};

BOOST_FUSION_ADAPT_STRUCT(strct, (int, a) (double, b) (char, c))

using namespace std;
using namespace boost::fusion;
using namespace boost::lambda;

int main()
{
    strct s{12, 3.14, 'K'};
    print_struct(s);
    filter_view<strct, boost::is_integral<boost::mpl::arg<1>>> v{s};
    for_each(v, cout << _1 << '\n');
}
