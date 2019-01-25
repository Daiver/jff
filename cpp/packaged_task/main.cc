#include <iostream>
#include <cmath>
#include <thread>
#include <future>
#include <functional>
 
// unique function to avoid disambiguating the std::pow overload set
int f(int x, int y) { return std::pow(x,y); }
 
void task_lambda()
{
    std::packaged_task<int(int,int)> task([](int a, int b) {
        return std::pow(a, b); 
    });
    std::future<int> result = task.get_future();
 
    task(2, 9);
 
    std::cout << "task_lambda:\t" << result.get() << '\n';
}
 
void task_bind()
{
    std::packaged_task<int()> task(std::bind(f, 2, 11));
    std::future<int> result = task.get_future();
 
    task();
 
    std::cout << "task_bind:\t" << result.get() << '\n';
}
 
void task_thread()
{
    std::packaged_task<int(int,int)> task(f);
    std::future<int> result = task.get_future();
 
    std::thread task_td(std::move(task), 2, 10);
    task_td.join();
 
    std::cout << "task_thread:\t" << result.get() << '\n';
}

int treeRecur(int i, int d){
    std::cout << "lvl: " << d << " " << i << "\n";
    if(d > 105) return i;
    std::packaged_task<int(int, int)> task2([](int a, int b) {
        return treeRecur(a,b);
    });
    std::future<int> result2 = task2.get_future();
     std::packaged_task<int(int, int)> task([](int a, int b) {
        return treeRecur(a,b);
    });
    std::future<int> result = task.get_future();
    // std::thread th (std::move(tsk),10,0);
    task(i + 5, d + 1);
    task2(i + 5, d + 1);
    return result.get() + result2.get();
}

void task_lambda2()
{
    std::packaged_task<int(int,int)> task([](int a, int b) {
        return treeRecur(a,b);
    });
    std::future<int> result = task.get_future();
 
    task(1, 1);
 
    std::cout << "task_lambda:\t" << result.get() << '\n';
}
 
int main()
{
    task_lambda2();
}
