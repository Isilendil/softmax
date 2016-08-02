///////////////////////////////////////////////////////
//softmax

#ifndef LIBLINEAR_TEST_SOLVER_SOFTMAX_H
#define LIBLINEAR_TEST_SOLVER_SOFTMAX_H

#include "linear.h"


class Solver_SOFTMAX
{
	public:
	  Solver_SOFTMAX(const problem *prob, int nr_class, double C, double eps = 0.00001, int max_iter = 100, double eta = 0.00001);
		//eta = 0.000001 for pendigits
		//eta = 0.01 for real-sim will generate NaN error
		~Solver_SOFTMAX();
		void Solve(double *w, double *obj);

	int max_iter;

	private:
	void solve_sub_problem();
	double compute_obj(double *w);
	int w_size;
	int l;
	int nr_class;
	double eps;
	double C;
	double eta;

	const problem *prob;
	int n;
};
#endif //LIBLINEAR_TEST_SOLVER_SOFTMAX_H

///////////////////////////////////////////////////////
