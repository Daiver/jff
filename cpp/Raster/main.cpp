#include <QCoreApplication>

#include <QString>
#include <QVector>
#include <QVector2D>
#include <QVector3D>
#include <QMatrix4x4>
#include <QImage>
#include <QPainter>

#include "Eigen/Dense"

#include "objimportexport.h"

QVector2D barCoords(const QVector2D &p1, const QVector2D &p2, const QVector2D &p3, const QVector2D &p)
{
    Eigen::Matrix2f T;
    T(0, 0) = p1.x() - p3.x();
    T(0, 1) = p2.x() - p3.x();
    T(1, 0) = p1.y() - p3.y();
    T(1, 1) = p2.y() - p3.y();
    Eigen::Vector2f r;
    r[0] = p.x() - p3.x();
    r[1] = p.y() - p3.y();
    Eigen::Vector2f res = T.inverse() * r;
    return QVector2D(res[0], res[1]);
}

QVector3D transformVertex(const QMatrix4x4 &transformation, const QVector3D &vertex)
{
    return QVector3D(transformation * QVector4D(vertex, 1));
}

void rasterizeTriangle(
        const QVector2D &p1,
        const float z1,
        const QVector2D &p2,
        const float z2,
        const QVector2D &p3,
        const float z3,
        QImage &img)
{
    float left  = img.width();
    float right = 0;
    float up    = img.height();
    float down  = 0;

    left = (p1.x() < left) ? p1.x() : left;
    left = (p2.x() < left) ? p2.x() : left;
    left = (p3.x() < left) ? p3.x() : left;

    right = (p1.x() > right) ? p1.x() : right;
    right = (p2.x() > right) ? p2.x() : right;
    right = (p3.x() > right) ? p3.x() : right;

    up = (p1.y() < up) ? p1.y() : up;
    up = (p2.y() < up) ? p2.y() : up;
    up = (p3.y() < up) ? p3.y() : up;

    down = (p1.y() > down) ? p1.y() : down;
    down = (p2.y() > down) ? p2.y() : down;
    down = (p3.y() > down) ? p3.y() : down;

    left  = round(left);
    right = round(right);
    up    = round(up);
    down  = round(down);

    for(int x = left; x <= right; ++x){
        for(int y = up; y <= down; ++y){
            const QVector2D p    = QVector2D(x, y);
            const QVector2D bars = barCoords(p1, p2, p3, p);
            const float z = bars.x() * z1 + bars.y() * z2 + (1.0 - bars.x() - bars.y()) * z3;
            const float zToRender = z * 255;

            if(zToRender < 0 || zToRender > 255)
                continue;
            const float pixVal = qRed(img.pixel(p.toPoint()));
            if(pixVal > zToRender)
                continue;
            if(bars.x() >= 0 && bars.x() <= 1.0 && bars.y() >= 0 && bars.y() <= 1.0){
                img.setPixel(p.toPoint(), qRgba(zToRender, zToRender, zToRender, 255));
            }
        }
    }
}

QVector2D projectVertexOntoPlane(const QSize &size, const QVector3D vertex, float *projectedZ = NULL)
{
    if(projectedZ != NULL)
        *projectedZ = vertex.z();
    return QVector2D(size.width() * vertex.x(), size.height() - size.height() * vertex.y());
}

QImage renderDepth(const QSize &size, const QMatrix4x4 &transformation, const QVector<QVector3D> &vertices, const QVector<int> &polygonStarts, const QVector<int> &polygonIndices)
{
    QImage res(size, QImage::Format_RGBA8888);
    res.fill(Qt::black);

    const int nPolygons = polygonStarts.size() - 1;
    int polygonOffset = 0;
    for(int polInd = 0; polInd < nPolygons; ++polInd){
        const int nVerticesInPol = polygonStarts[polInd + 1] - polygonStarts[polInd];
        Q_ASSERT(nVerticesInPol == 3 || nVerticesInPol == 4);

        const QVector3D v1 = transformVertex(transformation, vertices[polygonIndices[polygonOffset + 0]]);
        float z1 = 0;
        const QVector2D p1 = projectVertexOntoPlane(size, v1, &z1);

        const QVector3D v2 = transformVertex(transformation, vertices[polygonIndices[polygonOffset + 1]]);
        float z2 = 0;
        const QVector2D p2 = projectVertexOntoPlane(size, v2, &z2);

        const QVector3D v3 = transformVertex(transformation, vertices[polygonIndices[polygonOffset + 2]]);
        float z3 = 0;
        const QVector2D p3 = projectVertexOntoPlane(size, v3, &z3);

        if(nVerticesInPol == 3){
            rasterizeTriangle(p1, z1, p2, z2, p3, z3, res);
        } else if(nVerticesInPol == 4){
            const QVector3D v4 = transformVertex(transformation, vertices[polygonIndices[polygonOffset + 3]]);
            float z4 = 0;
            const QVector2D p4 = projectVertexOntoPlane(size, v4, &z4);

            rasterizeTriangle(p1, z1, p2, z2, p3, z3, res);
            rasterizeTriangle(p3, z3, p4, z4, p1, z1, res);
        }

        polygonOffset += nVerticesInPol;
    }

    return res;
}

int main()
{
    const QString fname = "/home/daiver/Teapot.obj";
    QVector<QVector3D> vertices;
    QVector<int> polygonStarts;
    QVector<int> polygonIndices;
    ObjImportExport::readObj(fname, vertices, polygonStarts, polygonIndices);
    for(int i = 0; i < 72; ++i){
        QMatrix4x4 trans;
        trans.translate(0.45, 0.3);
        trans.scale(0.6);
//        const float targetAngle = i * 5 + 2;
        const float targetAngle = i * 5;
        qDebug() << targetAngle;
        trans.rotate(QQuaternion::fromAxisAndAngle(QVector3D(0, 1, 0), targetAngle));
//        const QSize imgSize(32, 32);
        const QSize imgSize(64, 64);
//        const QSize imgSize(128, 128);
        const QImage res = renderDepth(imgSize, trans, vertices, polygonStarts, polygonIndices);
//        const QString destDir = "/home/daiver/dump/test/";
        const QString destDir = "/home/daiver/dump/train/";
        res.save(destDir + QString("res_%1.png").arg(i));
    }
    return 0;
}

