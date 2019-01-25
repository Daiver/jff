#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "ceres/local_parameterization.h"
#include "glog/logging.h"
#include "gflags/gflags.h"

#include <stdio.h>

int main()
{
    double point[3] = {2, 0, 0};
    double angleAxis[3] = {0, -M_PI/2 * 1, 0};
    double p[3];
    ceres::AngleAxisRotatePoint(angleAxis, point, p);
    //QuaternionParametrization rotation;

    printf("%f %f %f\n", p[0], p[1], p[2]);

    return 0;
}
