#include <stdio.h>
#include <memory>

struct Song
{
    int i, j;
    Song(int i, int j) : i(i), j(j) {}
};

int main()
{
    auto p = std::make_shared<Song>(5, 17);
    printf("%d %d\n", p->i, p->j);
    return 0;
}
