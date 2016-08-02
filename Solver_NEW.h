///////////////////////////////////////////////////////
//dual coordinate descent for softmax regression

#ifndef LIBLINEAR_TEST_SOLVER_NEW_H
#define LIBLINEAR_TEST_SOLVER_NEW_H

#include "linear.h"


class Solver_NEW
{
	public:
	  Solver_NEW(const problem *prob, int nr_class, double C, double eps = 0.00001, int max_iter = 100, double eta_value = 0.5, int max_trial = 20, double initial = 1e-10, int max_inner_iter = 20, int update_index_size = 4);
		~Solver_NEW();
		void Solve(double *w, double *obj);

	int max_iter;

	private:
	void solve_sub_problem(double a, double b, double c, double *pd);
	double compute_obj(double *w);

	void adjust_stepsize(double *d, double *upper, double *lower);
  //double compute_dual_dif(double *w);

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

	int update_index_size;
	
	double *norm_list;
	
	//double *delta_f;
};
#endif //LIBLINEAR_TEST_SOLVER_NEW_H

///////////////////////////////////////////////////////
