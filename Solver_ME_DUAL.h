///////////////////////////////////////////////////////
//dual coordinate descent for maximum entropy models

#ifndef LIBLINEAR_TEST_SOLVER_ME_DUAL_H
#define LIBLINEAR_TEST_SOLVER_ME_DUAL_H

#include "linear.h"


class Solver_ME_DUAL
{
	public:
	  Solver_ME_DUAL(const problem *prob, int nr_class, double C, double eps = 0.01, int max_iter = 100, double eta_value = 0.5, int max_trial = 20, double initial = 1e-10, int max_inner_iter = 100);
		~Solver_ME_DUAL();
		void Solve(double *w, double *obj);

	int max_iter;

	private:
	void solve_sub_problem(double a, double b, double c_1, double c_2, double *z_1, double *z_2);
	double compute_obj(double *w);
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
	
	double *norm_list;
	
	//double *delta_f;
};
#endif //LIBLINEAR_TEST_SOLVER_ME_DUAL_H

///////////////////////////////////////////////////////
