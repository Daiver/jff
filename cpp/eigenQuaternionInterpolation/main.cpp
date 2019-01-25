#include <Eigen/Dense>
#include <vector>
#include <iostream>

Eigen::Vector3f vector3f(const float x, const float y, const float z)
{
    Eigen::Vector3f res;
    res << x, y, z;
    return res;
}

Eigen::Vector4f quatToVector4f(const Eigen::Quaternion<float> &quat)
{
    Eigen::Vector4f res;
    res << quat.x(), quat.y(), quat.z(), quat.w();
    return res;
}

Eigen::Vector3f transformWeighted(
        const std::vector<Eigen::Quaternion<float>> &rotations,
        const std::vector<float> &weights,
        const Eigen::Vector3f &point)
{
    using namespace Eigen;
    Vector3f res = vector3f(0, 0, 0);
    for(int i = 0; i < weights.size(); ++i)
        res += weights[i] * (rotations[i].matrix() * point);
    return res;
}

std::pair<float, Eigen::Vector4f> selectDominantEigen(
        const Eigen::Vector4cf &evalues,
        const Eigen::Matrix4cf &evectors)
{
    std::pair<float, Eigen::Vector4f> res = std::make_pair(evalues[0].real(), evectors.real().col(0));
    for(int i = 0; i < 4; ++i){
        if(evalues[i].real() > res.first){
            res = std::make_pair(evalues[i].real(), evectors.real().col(i));
        }
    }
    return res;
}

Eigen::Quaternion<float> interpolateQuaternions(
        const std::vector<Eigen::Quaternion<float>> &rotations,
        const std::vector<float> &weights)
{
    using namespace Eigen;
    using namespace std;
    Matrix4f problemMat = Matrix4f::Zero();
    for(int i = 0; i < weights.size(); ++i){
        Eigen::Vector4f tmp = quatToVector4f(rotations[i]);
        problemMat += weights[i] * (tmp * tmp.transpose());
    }
    cout << "Problem mat" << endl << problemMat << endl;
    EigenSolver<Matrix4f> solver(problemMat);
    auto res = selectDominantEigen(solver.eigenvalues(), solver.eigenvectors());
    cout << "The eigenvalues of M are:" << endl 
         << solver.eigenvalues() << endl;
    cout << "The matrix of eigenvectors, V, is:" << endl 
         << solver.eigenvectors() << endl << endl;
    cout << res.first << endl;
    return Quaternion<float>(res.second[3], res.second[0], res.second[1], res.second[2]);
}

void test1()
{
    using namespace Eigen;
    using namespace std;
    //Quaternion<float> rot(AngleAxis<float>(3.14, vector3f(1, 0, 0)));
    std::vector<Quaternion<float>> rotations = {
        Quaternion<float>(AngleAxis<float>( M_PI,       vector3f(0, 0, 1).normalized())),
        Quaternion<float>(AngleAxis<float>( M_PI*2.0/3, vector3f(1, 1, 1).normalized())),
        Quaternion<float>(AngleAxis<float>( M_PI/3.0,   vector3f(1, 0, 1).normalized())),
        Quaternion<float>(AngleAxis<float>(-M_PI/2.0,   vector3f(1, 1, 0).normalized()))
    };
    std::vector<float> weights = {
        0.1, 
        0.2, 
        0.3,
        0.4
    };
    for(int i = 0; i < rotations.size(); ++i)
        rotations[i].normalize();

    auto interpolated = interpolateQuaternions(rotations, weights).normalized();

    cout << "interpolated" << endl << quatToVector4f(interpolated) << endl;
    Vector3f point = vector3f(1, 2, 3);
    cout << "p1" << endl << transformWeighted(rotations, weights, point) << endl;
    cout << "p2" << endl << interpolated * point << endl;
    cout << "p3" << endl << rotations[0].slerp(weights[1], rotations[1]) * point << endl;
}

void test2()
{
    using namespace Eigen;
    using namespace std;
    
    Matrix4f m;
    m << 1, 2, 0, 5,
         2, 4, 7, 1,
         1, 7, 1, 0,
         5, 1, 0, 4;
    EigenSolver<Matrix4f> solver(m);
    cout << "The eigenvalues of M are:" << endl 
         << solver.eigenvalues() << endl;
    cout << "The matrix of eigenvectors, V, is:" << endl 
         << solver.eigenvectors() << endl << endl;
    auto res = selectDominantEigen(solver.eigenvalues(), solver.eigenvectors());
    cout << "best" << endl << res.first << endl << res.second << endl;
}

int main()
{
    test1();
    //test2();
    //std::cout << quatToVector4f(Eigen::Quaternion<float>(1, 0, 0, 0)) << std::endl;
    return 0;
}
