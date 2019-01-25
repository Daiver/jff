#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include <string>
#include <sstream>
#include <iostream>
#include <vector>

using namespace tbb;
using namespace std;



class MD5Calculate
{
public:
  //////////////////////////////////////////////////////////////////////////
  // Задача, которая будет параллелиться с помощью TBB
  // входным параметром является некий диапазон,
  // в нашем случае это "кусок" из заданного диапазона чисел
  //////////////////////////////////////////////////////////////////////////

  MD5Calculate(vector<int> *res){
    this->res = res;
  }

  vector<int> *res;
  void operator() (const blocked_range<int>& range) const 
  {
    string md5;
    stringstream stream;
    for(int i = range.begin(); i != range.end(); i++) 
    {
        int cur = 1;
        for(int j = i; j < 1000000; j++){
            if(j % (i+1) == 0 && cur < j){
                cur = j;
            }
        }
        this->res->at(i) = cur;
    }
  }
};

int main()
{
  task_scheduler_init init;

  //////////////////////////////////////////////////////////////////////////
  // В данном примере пароль это число, лежащее в пределах
  // от 8000000 до 8999999
  //////////////////////////////////////////////////////////////////////////
  vector<int> res(501, -1);
  parallel_for(blocked_range<int>(0, 500), MD5Calculate(&res));
  for(int i = 0; i < res.size(); i++){
    cout << i << " " << res[i] << "\n";
  }
  return 0;
}
