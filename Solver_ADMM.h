///////////////////////////////////////////////////////
//dual coordinate descent for softmax regression

#ifndef LIBLINEAR_TEST_SOLVER_ADMM_H
#define LIBLINEAR_TEST_SOLVER_ADMM_H

#include "linear.h"
#include "Function_SOFTMAX.h"

class Solver_ADMM
{
	public:
	  Solver_ADMM(const problem *prob, int nr_class, double C, double eps = 1e-10, \
	   int max_iter = 1000, \
        double initial = 1e-10, int max_inner_iter = 10, int max_newton_iter = 1);
		~Solver_ADMM();
    void Solve(double *w, double *obj, Function_SOFTMAX *func);

	int max_iter;

	private:
	//void solve_sub_problem();
	double compute_obj(double *w);

  //double compute_dual_dif(double *w);

	int w_size;
	int l;
	int nr_class;
	double eps;
	double C;


	const problem *prob;
	int n;

	int alpha_size;
	double *alpha;


	int max_inner_iter;
    int max_newton_iter;


	double *norm_list;

    double pi;
	
};
#endif //LIBLINEAR_TEST_SOLVER_ADMM_H

///////////////////////////////////////////////////////
