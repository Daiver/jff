#ifndef MYPROBLEM1_H
#define MYPROBLEM1_H

#include <iostream>
#include <QtGlobal>

#include "IpTNLP.hpp"

class MyProblem1 : public Ipopt::TNLP
{
public:
    const int nVars = 1;
    virtual bool get_nlp_info(Ipopt::Index& n, Ipopt::Index& m, Ipopt::Index& nnz_jac_g,
                              Ipopt::Index& nnz_h_lag, IndexStyleEnum& index_style)
    {
        n = nVars;
        m = 0;
        nnz_jac_g = 0;
        nnz_h_lag = 1;
        index_style = IndexStyleEnum::C_STYLE;
        return true;
    }

    virtual bool get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u,
                                 Ipopt::Index m, Ipopt::Number* g_l, Ipopt::Number* g_u)
    {
        Q_UNUSED(m);
        Q_UNUSED(g_l);
        Q_UNUSED(g_u);
        for(int i = 0; i < n; ++i){
            x_l[i] = -1e19; //nlp_lower_bound_inf
            x_u[i] = 1e19; //nlp_upper_bound_inf
        }
        return true;
    }

    virtual bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number* x,
                                    bool init_z, Ipopt::Number* z_L, Ipopt::Number* z_U,
                                    Ipopt::Index m, bool init_lambda,
                                    Ipopt::Number* lambda)
    {
        Q_ASSERT(nVars == n);
        Q_ASSERT(0 == m);
        Q_ASSERT(!init_z);
        Q_ASSERT(!init_lambda);
        Q_UNUSED(z_L);
        Q_UNUSED(z_U);
        Q_UNUSED(m);
        Q_UNUSED(lambda);

        if(init_x){
            x[0] = 0.0;
        }
        return true;
    }

    virtual bool eval_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                        Ipopt::Number& obj_value)
    {
        Q_ASSERT(nVars == n);
        Q_UNUSED(new_x);
        const Ipopt::Number residual = (5 * x[0] - 3);
        obj_value = residual * residual;
        return true;
    }

    virtual bool eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                             Ipopt::Number* grad_f)
    {
        Q_ASSERT(nVars == n);
        const Ipopt::Number residual = (5 * x[0] - 3);
        grad_f[0] = 2 * 10 * residual;
        return true;
    }

    virtual bool eval_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                        Ipopt::Index m, Ipopt::Number* g)
    {
        Q_ASSERT(nVars == n);
        return true;
    }

    virtual bool eval_jac_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                            Ipopt::Index m, Ipopt::Index nele_jac, Ipopt::Index* iRow,
                            Ipopt::Index *jCol, Ipopt::Number* values)
    {
        return true;
    }

    virtual void finalize_solution(Ipopt::SolverReturn status,
                                   Ipopt::Index n, const Ipopt::Number* x, const Ipopt::Number* z_L, const Ipopt::Number* z_U,
                                   Ipopt::Index m, const Ipopt::Number* g, const Ipopt::Number* lambda,
                                   Ipopt::Number obj_value,
                                   const Ipopt::IpoptData* ip_data,
                                   Ipopt::IpoptCalculatedQuantities* ip_cq)
    {
        std::cout << "f(x*) = " << obj_value << std::endl;
        std::cout << "X final ";
        for(int i = 0; i < n; ++i)
            std::cout << x[i] << " ";
        std::cout << std::endl;
    }
};


#endif // MYPROBLEM1_H
