
#ifndef LIBLINEAR_TEST_FUNCTION_SOFTMAX_H
#define LIBLINEAR_TEST_FUNCTION_SOFTMAX_H

#include "linear.h"

class Function_SOFTMAX {

public:

    Function_SOFTMAX(const problem *prob, const problem *prob_t, int nr_class, double C = 1.0);
    ~Function_SOFTMAX();

    double obj_primal(double *w);
    double obj_dual(double *alpha, double *w);

    void grad(double *w, double *grad, double *p_grad_norm);

    double testing(double *w);

private:

    double C;

    const problem *prob;
    const problem *prob_t;

    int nr_class;
    int l;
    int l_t;
    int n;
    int w_size;

};


#endif //LIBLINEAR_TEST_FUNCTION_SOFTMAX_H
