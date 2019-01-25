#ifndef STUMPTESTS_H
#define STUMPTESTS_H

#include <QTest>

class StumpTests : public QObject
{
    Q_OBJECT
public:
    StumpTests();

private slots:
    void testSortIndicesByFeature01();
    void testSortIndicesByFeature02();
    void testSortIndicesByFeature03();

    void testTargetLossInfoMSE01();
    void testTargetLossInfoMSE02();
    void testTargetLossInfoMSE03();

    void testFindBestSplitAcrossFeature01();
};

#endif // STUMPTESTS_H
