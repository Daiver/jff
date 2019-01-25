#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/task_group.h"

#include <string>
#include <sstream>
#include <iostream>
#include <vector>

using namespace tbb;
using namespace std;

int fibTBB(int n) {
    if (n == 1) return 1;
    if (n == 0) return 1;
    int x, y;
    tbb::task_group g;
    g.run([&]{x=fibTBB(n-1);}); // spawn a task
    g.run([&]{y=fibTBB(n-2);}); // spawn another task
    g.wait();                // wait for both tasks to complete
    return x+y;
}

int main(){
    std::cout << fibTBB(8);
    return 0;
}
