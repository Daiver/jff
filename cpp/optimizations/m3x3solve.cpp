#include <stdio.h>
#include <math.h>

class QVector3D
{
    public:
        QVector3D(float x, float y, float z):
            x(x), y(y), z(z)
            {
            };

        float x;
        float y;
        float z;
};

template<int N>
inline double item(const double *const a, int i, int j)
{
    return a[N*i + j];
}

QVector3D solve3x3(const double *const a, const QVector3D &b)
{
    const int N = 3;
    double coeffs[N * N];
    coeffs[0] = item<N>(a, 1, 1)*item<3>(a, 2, 2) - item<3>(a, 1, 2)*item<3>(a, 2, 1);
    coeffs[1] = item<N>(a, 0, 2)*item<3>(a, 2, 1) - item<3>(a, 0, 1)*item<3>(a, 2, 2);
    coeffs[2] = item<N>(a, 0, 1)*item<3>(a, 1, 2) - item<3>(a, 0, 2)*item<3>(a, 1, 1);
    coeffs[3] = item<N>(a, 1, 2)*item<3>(a, 2, 0) - item<3>(a, 1, 0)*item<3>(a, 2, 2);
    coeffs[4] = item<N>(a, 0, 0)*item<3>(a, 2, 2) - item<3>(a, 0, 2)*item<3>(a, 2, 0);
    coeffs[5] = item<N>(a, 0, 2)*item<3>(a, 1, 0) - item<3>(a, 0, 0)*item<3>(a, 1, 2);
    coeffs[6] = item<N>(a, 1, 0)*item<3>(a, 2, 1) - item<3>(a, 1, 1)*item<3>(a, 2, 0);
    coeffs[7] = item<N>(a, 0, 1)*item<3>(a, 2, 0) - item<3>(a, 0, 0)*item<3>(a, 2, 1);
    coeffs[8] = item<N>(a, 0, 0)*item<3>(a, 1, 1) - item<3>(a, 0, 1)*item<3>(a, 1, 0);

    double det = (a[0]*coeffs[0] + a[1]*coeffs[3] + a[2]*coeffs[6]);
    printf("det %f\n", det);
    if(det != 0)
        for(int i = 0; i < N*N; ++i){
            coeffs[i] /= (det);
        }
    else{
        printf("Zero det\n");
    }

    QVector3D res(
            coeffs[0]*b.x + coeffs[1]*b.y + coeffs[2]*b.z, 
            coeffs[3]*b.x + coeffs[4]*b.y + coeffs[5]*b.z, 
            coeffs[6]*b.x + coeffs[7]*b.y + coeffs[8]*b.z
            );
    return res;
}

int main(){
    double a[9] = {1,2,3,4,5,6,7,8,1};
    QVector3D x = solve3x3(a, QVector3D(1,2,3));
    printf("\n%f %f %f", x.x, x.y, x.z);
    return 0;
}
