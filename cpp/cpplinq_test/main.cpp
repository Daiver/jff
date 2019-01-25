#include <iostream>
#include <vector>
#include "cpplinq.hpp"

int computes_a_sum ()
{
}

int main()
{
    using namespace cpplinq;    
    std::vector<int> ints = {3,1,4,1,5,9,2,6,5,4};
    //int ints[] = {3,1,4,1,5,9,2,6,5,4};

    // Computes the sum of all even numbers in the sequence above
    auto res = 
            from (ints)
            //from_array (ints)
        >>  where ([](int i) {return i%2 ==0;})     // Keep only even numbers
        >>  select([](auto i){return std::to_string(i);})
        //>>  count ()                                  // Sum remaining numbers
        //>>  sum ()                                  // Sum remaining numbers
        ;
    std::cout << (res >> to_vector())[2] << std::endl;
	return 0;
}
