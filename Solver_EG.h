///////////////////////////////////////////////////////
//exponentiated gradient

#ifndef LIBLINEAR_TEST_SOLVER_EG_H
#define LIBLINEAR_TEST_SOLVER_EG_H

#include "linear.h"
#include "Function_SOFTMAX.h"

class Solver_EG
{
	public:
	  Solver_EG(const problem *prob, int nr_class, double C, double eps = 0.01, int max_iter = 200, double eta_value = 0.5, int max_trial = 20, double initial = 1e-8);
		~Solver_EG();
		void Solve(double *w, double *obj, Function_SOFTMAX *func);

	int max_iter;

	private:
	void solve_sub_problem();
	double compute_obj(double *w);
  double compute_dual_dif(double *v);

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

	double *norm_list;
	//double *delta_f;
};
#endif //LIBLINEAR_TEST_SOLVER_EG_H

///////////////////////////////////////////////////////
