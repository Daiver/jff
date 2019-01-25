#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <type_traits>

template<typename ScalarType, typename ItemIndType>
class StaticSumFunctor
{
public:
    static ScalarType apply(const ScalarType *vec) 
    {
        return vec[ItemIndType::value] + StaticSumFunctor<ScalarType, std::integral_constant<int, ItemIndType::value - 1> >::apply(vec);
    }
};

template<typename ScalarType>
class StaticSumFunctor<ScalarType, std::integral_constant<int, 0> >
{
public:
    static ScalarType apply(const ScalarType *vec) 
    {
        return vec[0];
    }
};



/*template<typename ScalarType, typename ItemIndType>*/
//ScalarType staticSum(const ScalarType *vec)
//{
    //return vec[ItemIndType::value] + staticSum<ScalarType, std::integral_constant<int, ItemIndType::value - 1> >(vec);
//}

//typedef std::integral_constant<int, 0> zero;

//template<typename ScalarType>
//ScalarType staticSum<ScalarType, zero >(const ScalarType *vec)
//{
    //return vec[0];
//}

int main()
{
    //typedef float ScalarType;
    //typedef double ScalarType;
    //const ScalarType vec[] = {1, 2, 3, 4, 5, 6};
    typedef std::string ScalarType;
    const ScalarType vec[] = {"1", "2", "3", "4"};
    const ScalarType *vecPtr = vec;
    //std::cout << vec[0] + vec[1] + vec[2];
    std::cout << StaticSumFunctor<ScalarType, std::integral_constant<int, 3> >::apply(vecPtr)  << std::endl;
    //std::cout << staticSum<ScalarType, std::integral_constant<int, 5> >(vecPtr)  << std::endl;
    return 0;	    
}
