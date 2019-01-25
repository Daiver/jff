#include <QCoreApplication>

#include "Eigen/Core"
#include "range/v3/all.hpp"

#include "stumptests.h"

void runTests()
{
    StumpTests sTests;
    QTest::qExec(&sTests);
}

int main(int argc, char *argv[])
{
    runTests();
    return 0;
}
