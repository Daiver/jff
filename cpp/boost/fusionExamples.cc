
#include <boost/fusion/sequence.hpp>
#include <boost/fusion/algorithm.hpp>
#include <iostream>

using namespace std;
using namespace boost;

int main()
{
    typedef fusion::vector<int,float,char> sequence_t;
    sequence_t sequence(42, 3.14f, 'c');
     
    cout << fusion::size(sequence) << endl;
    cout << fusion::at_c<0>(sequence) << endl;
    cout << fusion::at_c<1>(sequence) << endl;
    cout << fusion::at_c<2>(sequence) << endl;
     
    //we can also get compile time information
    int array[fusion::result_of::size<sequence_t>::value];
     
    boost::remove_reference<
            fusion::result_of::at_c<sequence_t, 1>::type
         >::type foo; // foo is a float
    return 0;
}
