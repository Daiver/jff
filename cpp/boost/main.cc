
#include <boost/lambda/lambda.hpp>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <map>
#include <functional>
#include <string>
#include <boost/hana.hpp>

using namespace boost::lambda;

int f(int i, int j){return i+j;}

int main()
{
    std::map<std::string, std::function<int(int,int)>> mp = {
        {"+", _1 + _2}, {"-", _1 - _2}
    };
    printf("%d\n", mp["-"](1,2));
    printf("%d\n", mp["+"](1,2));
    //static std::map<std::string,std::function<T(T,T)>> evalFunction;
    //auto ff = ([&](int a, int b){return a+b});
    //std::map<std::string, decltype(ff)> mp = {{"1", [&](int a, int b){return 1;}}};

    //typedef std::istream_iterator<int> in;
    //std::cout << "Type in any number: ";
    /*std::for_each(
        in(std::cin), in(), std::cout 
                << (_1 * 10) 
                << "\nType in another number: " );
    */
}
