///////////////////////////////////////////////////////
//dual coordinate descent for softmax regression

#ifndef LIBLINEAR_TEST_SOLVER_ALM_H
#define LIBLINEAR_TEST_SOLVER_ALM_H

#include "linear.h"
#include "Function_SOFTMAX.h"

class Solver_ALM
{
	public:
	  Solver_ALM(const problem *prob, int nr_class, double C, double eps = 1e-20, \
	   int max_iter = 200, \
        double initial = 1e-10, int max_inner_iter = 5, int max_newton_iter = 5);
		~Solver_ALM();
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
#endif //LIBLINEAR_TEST_SOLVER_ALM_H

///////////////////////////////////////////////////////
