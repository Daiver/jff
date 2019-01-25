#ifndef COMMONEIGENROUTINE_H
#define COMMONEIGENROUTINE_H

#include "Eigen/Core"

namespace CommonEigenRoutine {

template<typename MatrixTypeIn>
Eigen::Map<Eigen::Matrix<typename MatrixTypeIn::Scalar, -1, -1, Eigen::RowMajor>, Eigen::Aligned> 
    reshape(MatrixTypeIn &matrix, int rows, int cols);

template<typename MatrixTypeIn>
const Eigen::Map<Eigen::Matrix<
        typename MatrixTypeIn::Scalar, -1, -1, Eigen::RowMajor>, Eigen::Aligned> 
    reshape(const MatrixTypeIn &matrix, int rows, int cols);

template<typename Scalar>
Eigen::Map<Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>, Eigen::Aligned> 
    reshape(Scalar *pointer, int rows, int cols);

template<typename Scalar>
const Eigen::Map<Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>, Eigen::Aligned> 
    reshape(const Scalar *pointer, int rows, int cols);

template<typename Derived, typename Func>
void mapMut(const Func &func, Eigen::MatrixBase<Derived> &mat);

}







template<typename MatrixTypeIn>
Eigen::Map<Eigen::Matrix<typename MatrixTypeIn::Scalar, -1, -1, Eigen::RowMajor>, Eigen::Aligned> 
        CommonEigenRoutine::reshape(MatrixTypeIn &matrix, int rows, int cols)
{
    assert(matrix.rows() * matrix.cols() == rows * cols);
    return CommonEigenRoutine::reshape<typename MatrixTypeIn::Scalar>(matrix.data(), rows, cols);
}

template<typename Scalar>
Eigen::Map<Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>, Eigen::Aligned> 
        CommonEigenRoutine::reshape(Scalar *pointer, int rows, int cols)
{
    return Eigen::Map<Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>, Eigen::Aligned> (
            pointer, rows, cols) ;
}


template<typename MatrixTypeIn>
const Eigen::Map<Eigen::Matrix<typename MatrixTypeIn::Scalar, -1, -1, Eigen::RowMajor>, Eigen::Aligned> 
        CommonEigenRoutine::reshape(const MatrixTypeIn &matrix, int rows, int cols)
{
    assert(matrix.rows() * matrix.cols() == rows * cols);
    return CommonEigenRoutine::reshape<typename MatrixTypeIn::Scalar>(matrix.data(), rows, cols);
}

template<typename Scalar>
const Eigen::Map<Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>, Eigen::Aligned> 
        CommonEigenRoutine::reshape(const Scalar *pointer, int rows, int cols)
{
    return Eigen::Map<Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>, Eigen::Aligned> (
            pointer, rows, cols) ;
}

template<typename Derived, typename Func>
void CommonEigenRoutine::mapMut(const Func &func, Eigen::MatrixBase<Derived> &mat)
{
    for(int i = 0; i < mat.rows(); ++i)
        for(int j = 0; j < mat.cols(); ++j)
            mat(i, j) = func(mat(i, j));
}

#endif
