///////////////////////////////////////////////////////
//dual coordinate descent for softmax regression models

#ifndef LIBLINEAR_TEST_SOLVER_CD_DUAL_H
#define LIBLINEAR_TEST_SOLVER_CD_DUAL_H

#include "linear.h"
#include "Function_SOFTMAX.h"

class Solver_CD_DUAL
{
	public:
	  Solver_CD_DUAL(const problem *prob, int nr_class, double C, double eps = 0.01, int max_iter = 1000, \
	     double initial = 1e-10, int max_inner_iter = 5, int max_newton_iter = 20);
		~Solver_CD_DUAL();
		void Solve(double *w, double *obj, Function_SOFTMAX *func);

	int max_iter;

	private:
	void solve_sub_problem(double a, double b, double c_1, double c_2, double *z_1, double *z_2);
	double compute_obj(double *w);

	int w_size;
	int l;
	int nr_class;
	double eps;
	double C;

	double *eta;

	const problem *prob;
	int n;

	int alpha_size;
	double *alpha;
	double *alpha_new;

  int max_trial;
	double initial;

	int id_current;

	int max_inner_iter;
    int max_newton_iter;
	
	double *norm_list;
	
	//double *delta_f;
};
#endif //LIBLINEAR_TEST_SOLVER_CD_DUAL_H

///////////////////////////////////////////////////////
