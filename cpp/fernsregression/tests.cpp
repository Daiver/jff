#include <stdio.h>

#include "minitest.h"
#include "common.h"
#include "ferns.h"

void testRange01()
{
    DataSet ds = {
        {-1, 1, 2},
        {10, 5, 3},
        {0, 11, -9},
        {0, 3, 7}
    };
    std::vector<Range> res = {
        {-1, 10}, {1, 11}, {-9, 7}
    };
    auto ans = computeRangesOfFeatures(ds);
    ASSERT_EQ((int)ans.size(), (int)res.size());
    for(int i = 0; i < ans.size(); ++i){
        ASSERT_EQ(ans[i].first, res[i].first);
        ASSERT_EQ(ans[i].second, res[i].second);

    }
}

int main()
{
    RUN_TEST(testRange01)
    return 0;
}
