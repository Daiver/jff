
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <map>
#include <vector>
#include <functional>
#include <string>
#include <boost/hana.hpp>
#include "cpplinq.hpp"


using namespace boost::lambda;
using namespace cpplinq;    
using namespace std;    

std::string inc(const int i)
{
    return std::to_string(i);
}

int main()
{
    vector<int> vec = {1, 2, 3};
    //std::function<std::string(int)> f = &to_string;
    auto f = inc;
    auto res = 
           from(vec)
        >> select(f)
        >> to_vector();
    for(auto x : res)
        cout << x << " ";
    cout << endl;
}
