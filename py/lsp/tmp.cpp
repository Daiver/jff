
#include <stdio.h>
#include <vector>

#include "builtins.h"


auto f(auto x, auto y) 
{
    const auto a = builtins::opAdd(x,y);
    const auto b = builtins::opSub(x,y);
    const auto c = builtins::opMul(a,b);
    
    return c;
}

    
auto abs(auto x) 
{
    ;
    
    return (builtins::opGreaterThan(x,0)) ? (x) : (builtins::opSub(0,x));
}

    
int main()
{
    printf("Start....\n");
    builtins::println(abs(1));
    builtins::println(abs(builtins::opSub(0,2)));
    builtins::println(true);
    builtins::println(builtins::opGreaterThan(1,2));
    builtins::println(builtins::opEqual(0,1));
    builtins::println(builtins::opEqual(0,0));
    const auto d = 1;
    return 0;
}

