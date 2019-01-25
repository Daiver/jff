#include <iostream>
#include <string>

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


struct s {
    int a;
    double b;
    char c;
};

BOOST_FUSION_ADAPT_STRUCT(s, (int, a) (double, b) (char, c))

int main()
{
    print_struct(s{12, 3.14, 'K'});
}
