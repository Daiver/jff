#include <iostream>
#include <vector>
#include "Eigen/Dense"

void estimateRigidTransformation(
        const std::vector<Eigen::Vector3d> &source,
        const std::vector<Eigen::Vector3d> &target,
        const std::vector<Eigen::Vector3d> &normal
        ) 
{
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    typedef Eigen::Matrix<double, 6, 6> Matrix6d;
  
    Matrix6d ATA;
    Vector6d ATb;
    ATA.setZero ();
    ATb.setZero ();
  
    // Approximate as a linear least squares problem
    for(int i = 0; i < source.size(); ++i)
    {
  
      const float & sx = source[i][0];
      const float & sy = source[i][1];
      const float & sz = source[i][2];
      const float & dx = target[i][0];
      const float & dy = target[i][1];
      const float & dz = target[i][2];
      const float & nx = normal[i][0];
      const float & ny = normal[i][1];
      const float & nz = normal[i][2];
  
      double a = nz*sy - ny*sz;
      double b = nx*sz - nz*sx; 
      double c = ny*sx - nx*sy;
     
      //    0  1  2  3  4  5
      //    6  7  8  9 10 11
      //   12 13 14 15 16 17
      //   18 19 20 21 22 23
      //   24 25 26 27 28 29
      //   30 31 32 33 34 35
     
      ATA.coeffRef (0) += a * a;
      ATA.coeffRef (1) += a * b;
      ATA.coeffRef (2) += a * c;
      ATA.coeffRef (3) += a * nx;
      ATA.coeffRef (4) += a * ny;
      ATA.coeffRef (5) += a * nz;
      ATA.coeffRef (7) += b * b;
      ATA.coeffRef (8) += b * c;
      ATA.coeffRef (9) += b * nx;
      ATA.coeffRef (10) += b * ny;
      ATA.coeffRef (11) += b * nz;
      ATA.coeffRef (14) += c * c;
      ATA.coeffRef (15) += c * nx;
      ATA.coeffRef (16) += c * ny;
      ATA.coeffRef (17) += c * nz;
      ATA.coeffRef (21) += nx * nx;
      ATA.coeffRef (22) += nx * ny;
      ATA.coeffRef (23) += nx * nz;
      ATA.coeffRef (28) += ny * ny;
      ATA.coeffRef (29) += ny * nz;
      ATA.coeffRef (35) += nz * nz;
  
      double d = nx*dx + ny*dy + nz*dz - nx*sx - ny*sy - nz*sz;
      ATb.coeffRef (0) += a * d;
      ATb.coeffRef (1) += b * d;
      ATb.coeffRef (2) += c * d;
      ATb.coeffRef (3) += nx * d;
      ATb.coeffRef (4) += ny * d;
      ATb.coeffRef (5) += nz * d;
  
    }
    ATA.coeffRef (6) = ATA.coeff (1);
    ATA.coeffRef (12) = ATA.coeff (2);
    ATA.coeffRef (13) = ATA.coeff (8);
    ATA.coeffRef (18) = ATA.coeff (3);
    ATA.coeffRef (19) = ATA.coeff (9);
    ATA.coeffRef (20) = ATA.coeff (15);
    ATA.coeffRef (24) = ATA.coeff (4);
    ATA.coeffRef (25) = ATA.coeff (10);
    ATA.coeffRef (26) = ATA.coeff (16);
    ATA.coeffRef (27) = ATA.coeff (22);
    ATA.coeffRef (30) = ATA.coeff (5);
    ATA.coeffRef (31) = ATA.coeff (11);
    ATA.coeffRef (32) = ATA.coeff (17);
    ATA.coeffRef (33) = ATA.coeff (23);
    ATA.coeffRef (34) = ATA.coeff (29);
  
    // Solve A*x = b
    Matrix6d reg = Matrix6d::Identity();
    reg *= 1e-7;
    ATA += reg;
    
    Vector6d x = static_cast<Vector6d> (ATA.inverse () * ATb);
    std::cout << x << std::endl;
    
    // Construct the transformation matrix from x
}

Eigen::Vector3d vec3(const float x, const float y, const float z)
{
    Eigen::Vector3d res;
    res << x, y, z;
    return res;
}

int main()
{
    std::vector<Eigen::Vector3d> base = {
        vec3( 1, 0, 0), vec3(0, 0.98480773, 0.173648164), vec3(0, -0.173648164, 0.98480773),
        vec3(-1, 0, 0), vec3(0, -0.98480773, -0.173648164), vec3(0, 0.173648164, -0.98480773)
    };
    std::vector<Eigen::Vector3d> normals = {
        vec3( 1, 0, 0), vec3(0,  1, 0), vec3(0, 0,  1),
        vec3(-1, 0, 0), vec3(0, -1, 0), vec3(0, 0, -1)
    };
    std::vector<Eigen::Vector3d> targets = {
        vec3( 1, 0, 0), vec3(0,  1, 0), vec3(0, 0,  1),
        vec3(-1, 0, 0), vec3(0, -1, 0), vec3(0, 0, -1)
    };
    estimateRigidTransformation(base, targets, normals);
    return 0;
}
