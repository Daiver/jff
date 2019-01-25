#include <chrono>
#include <functional>
#include <ctime>
#include <iostream>
#include <vector>

#define N 2000

std::chrono::high_resolution_clock::duration measure(std::function<void()> f, int n = 100)
{
    auto begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++)
    {
        f();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return (end - begin) / n;
}

void c_style_sort(int *m, int n) 
{
    for (int i = 0; i < N - 1; i++)
        for (int j = i + 1; j < N; j++) {
            if (m[i] < m[j])
            {
                //int tmp = m[i];
                //m[i] = m[j];
                //m[j] = tmp;
                std::swap(m[i], m[j]);
            }
        }
}

void c_style_test()
{
    int* m = new int[N];

    for (int i = 0; i < N; i++)
    {
        m[i] = i;
    }
    c_style_sort(m, N);
    delete[] m;
}

void cpp_style_sort(std::array<int, N> &m)
{
    auto n = m.size();
    for (int i = 0; i < n-1; i++)
        for (int j = i + 1; j < n; j++) {
            if (m[i] < m[j])
            {
                int tmp = m[i];
                m[i] = m[j];
                m[j] = tmp;
            }
        }
}

void cpp_style_test()
{
    std::array<int, N> m;

    for (int i = 0; i < N; i++)
    {
        m[i] = i; 
    }
    cpp_style_sort(m);
}

void vector_sort(std::vector<int> &m)
{
    auto n = m.size();

    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++) {
            if (m[i] < m[j])
            {
                std::swap(m[i], m[j]);
                //int tmp = m[i];
                //m[i] = m[j];
                //m[j] = tmp;
            }
        }
}

void vector_test()
{
    std::vector<int> m;
    m.reserve(N);

    for (int i = 0; i < N; i++)
    {
        m.push_back(i);
    }
    vector_sort(m);
}

int main()
{
    auto c_time = measure(c_style_test, 1000);
    std::cout << "c test " << (c_time).count()/1000000.0 << std::endl;
    auto cpp_time = measure(cpp_style_test, 1000);
    std::cout << "cpp test " << (cpp_time).count()/1000000.0 << std::endl;
    auto vector_time = measure(vector_test, 1000);
    std::cout << "vec test " << (vector_time).count()/1000000.0 << std::endl;
    return 0;
}
