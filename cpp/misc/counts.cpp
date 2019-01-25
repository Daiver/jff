#include <stdio.h>
#include <string>
#include <vector>

std::vector<int> counts(const std::string &s)
{
    std::vector<int> res(256, 0);
    for(int i = 0; i < s.size(); ++i)
        res[s[i]]++;
    return res;
}

int main()
{
    std::string input = "eifjweijfiwjefiwjf";
    std::vector<int> res = counts(input);
    for(int i = 0; i < res.size(); ++i)
        if(res[i] > 0)
            printf("%c: %d ", i, res[i]);

}
