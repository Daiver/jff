#include "objimportexport.h"

#include <QFile>
#include <QTextStream>

void ObjImportExport::readObj(const QString &fname, QVector<QVector3D> &vertices, QVector<int> &polygonStarts, QVector<int> &polygonIndices)
{
    QFile inputFile(fname);
    if (!inputFile.open(QIODevice::ReadOnly)){
        Q_ASSERT(false);
    }

    int polygonsOffset = 0;
    QTextStream in(&inputFile);
    while (!in.atEnd()) {
        QString line = in.readLine();
        if(line.isEmpty())
            continue;
        QStringList tokens = line.split(" ", QString::SkipEmptyParts);
        if(tokens.size() == 0)
            continue;
        const bool isVertexDesc = tokens[0] == "v";
        const bool isFaceDesc   = tokens[0] == "f";
        if(!isFaceDesc && !isVertexDesc)
            continue;
        if(isVertexDesc){
            Q_ASSERT(tokens.size() == 4);
            vertices << QVector3D(tokens[1].toFloat(), tokens[2].toFloat(), tokens[3].toFloat());
            continue;
        }
        if(isFaceDesc){
            const int nItems = tokens.size() - 1;
            polygonStarts << polygonsOffset;
            polygonsOffset += nItems;
            for(int i = 0; i < nItems; ++i){
                const QStringList localTokens = tokens[i + 1].split("/");
                Q_ASSERT(localTokens.size() > 0);
                polygonIndices << (localTokens[0].toInt() - 1);
            }
        }

    }
    polygonStarts << polygonsOffset;
    inputFile.close();
}
