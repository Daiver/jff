#ifndef OBJIMPORTEXPORT_H
#define OBJIMPORTEXPORT_H

#include <QString>
#include <QVector>
#include <QVector3D>

namespace ObjImportExport {

void readObj(const QString &fname, QVector<QVector3D> &vertices, QVector<int> &polygonStarts, QVector<int> &polygonIndices);

}

#endif // OBJIMPORTEXPORT_H
