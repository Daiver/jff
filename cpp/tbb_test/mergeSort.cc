#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/task_group.h"

#include <vector>
#include <stdio.h>

template <typename T>
std::vector<T> merge(std::vector<T> &a, std::vector<T> &b){
    int aIdx = 0;
    int bIdx = 0;
    std::vector<T> res;
    res.reserve(a.size() + b.size());
    while(aIdx < a.size() && bIdx < b.size()) {

        if(a[aIdx] < b[bIdx]){
            res.push_back(a[aIdx]);
            ++aIdx;
        }else{
            res.push_back(b[bIdx]);
            ++bIdx;
        }
    }
    for(int i = aIdx; i < a.size(); ++i)
        res.push_back(a[i]);
    for(int i = bIdx; i < b.size(); ++i)
        res.push_back(b[i]);
    return res;
}

template <typename T>
std::vector<T> mergeSort(std::vector<T> &vec){
    if(vec.size() == 1) return {vec[0]};
    if(vec.size() == 2){
        if(vec[0] > vec[1]) return {vec[1], vec[0]};
        else                return {vec[0], vec[1]};
    }

    //Splitting vector 
    int half_size = vec.size() / 2;
    std::vector<T> split_lo(vec.begin(), vec.begin() + half_size);
    std::vector<T> split_hi(vec.begin() + half_size, vec.end());

    //Sorting every part
    std::vector<T> sorted_split_lo, sorted_split_hi;
    tbb::task_group g;
    g.run([&]{sorted_split_lo = mergeSort(split_lo);});
    g.run([&]{sorted_split_hi = mergeSort(split_hi);});
    g.wait();
    //sorted_split_lo = mergeSort(split_lo);
    //sorted_split_hi = mergeSort(split_hi);

    //Merging two parts
    return merge(sorted_split_lo, sorted_split_hi);
}

int main(){

    std::vector<int> vec = {3,2,1, 5, 10, 7, 1};
    std::vector<int> vec2 = mergeSort(vec);
    
    for(auto x : vec2)
        printf("%d ", x);
    printf("\n");
    return 0;
}
