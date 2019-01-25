#include <stdio.h>
#include <vector>
#include <iostream>
#include <chrono>

template<typename TimeT = std::chrono::milliseconds>
struct measure
{
    template<typename F, typename ...Args>
    static typename TimeT::rep execution(F&& func, Args&&... args)
    {
        auto start = std::chrono::steady_clock::now();
        std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
        auto duration = std::chrono::duration_cast< TimeT> 
                            (std::chrono::steady_clock::now() - start);
        return duration.count();
    }
};

void test()
{
    const int N = 1;
    std::vector<int> res(N);
    for(int i = 0; i < N; ++i){
        res[i] = i * i;
        ++res[i];
    }
}

void test2()
{
    const int N = 1000000000;
    for(int i = 0; i < N; ++i) {
        test();
    }

}

int main()
{
    //std::cout << "simple test " << measure<>::execution(test) << std::endl;
    std::cout << "simple test " << measure<>::execution(test2) / 1000.0 << std::endl;
    return 0;
}
