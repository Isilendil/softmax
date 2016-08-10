//////////////////////////////////////////////
//dual coordinate descent for maximum entropy models
#include "Solver_ME_DUAL.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include <algorithm>
#include <time.h>

#include <iostream>
//////////////////////////////////////////////

Solver_ME_DUAL::Solver_ME_DUAL(const problem *prob, int nr_class, double C, double eps, int max_iter, double eta_value, int max_trial, double initial, int max_inner_iter)
{
	this->prob = prob;
	this->nr_class = nr_class;
	this->C = C;
	this->eps = eps;
	this->max_iter = max_iter;


	this->w_size = prob->n * nr_class;
	this->l = prob->l;
	this->n = prob->n;

  alpha_size = l * nr_class;
	alpha = new double[alpha_size];
	
	//alpha_new = new double[nr_class];

  this->max_trial = max_trial;
	this->initial = initial;


	this->max_inner_iter = max_inner_iter;


	norm_list = new double[l];
	for(int i = 0; i < l; i++)
	{
		norm_list[i] = 0;
	}
}

Solver_ME_DUAL::~Solver_ME_DUAL()
{
	delete [] alpha;
	//delete [] alpha_new;
	delete [] norm_list;
}

void Solver_ME_DUAL::Solve(double *w, double *obj, Function_SOFTMAX *func)
{
	/*
	int *perm = Malloc(int, l);
  double *obj = Malloc(double, max_iter+1);
	double *grad = Malloc(double, w_size);
	double *probability = Malloc(double, nr_class);
	double *norm_term = Malloc(double, nr_class);
	*/
	int *perm = new int[l];
	//double *obj = new double[max_iter+1];
	double *grad = new double[nr_class];

	double *grad_w = new double[w_size];

	double *output = new double[nr_class];
	//double *probability = new double[nr_class];
	//double *norm_term = new double[nr_class];
  //double sum = 0;
  //int id;
	double *v = new double[nr_class];
	double *z = new double[nr_class];
	double *G = new double[nr_class];



  double g_eps = 0.5;

	double timer = 0;
	double grad_norm;
	double primal;
	double dual;
	double accuracy;

  //initialize alpha
  for(int i = 0; i < alpha_size; i++)
	{
		alpha[i] = initial / (nr_class-1);
	}
	for(int id = 0; id < l; id++)
	{
		int yi = prob->y[id];
		alpha[id*nr_class+yi] = 1 - initial;
	}

  //initialize w
	//Eq. (41)
	//pre-calculate norm_list
	for(int i = 0; i < w_size; i++)
		w[i] = 0;
	
	for(int id = 0; id < l; id++)
	{
		feature_node *xi = prob->x[id];
		int yi = prob->y[id];

		while(xi->index != -1)
		{
			int index = xi->index - 1;
			double value = xi->value;

      //calculate norm_list
			norm_list[id] += value * value;

			for(int iter_class = 0; iter_class < nr_class; iter_class++)
			{
				if(iter_class != yi)
				{
				  w[index*nr_class+iter_class] += (-alpha[id*nr_class+iter_class] * value);
				}
			}
			w[index*nr_class+yi] += ((1-alpha[id*nr_class+yi]) * value);
			++xi;
		}
	}
		
  //compute objective
	obj[0] = compute_obj(w);


	int iter = 0;
	for(; iter < max_iter; iter++)
	{

		double start = clock();

		//permutation
		for(int i = 0; i < l; i++)
		{
			perm[i] = i;
		}



		for(int i = 0; i < l; i++)
		{
			int j = i + rand()%(l-i);
			//swap(perm[i], perm[j]);
			int temp = perm[i];
			perm[i] = perm[j];
			perm[j] = temp;
		}


		//two-level dual coordinate descent
		//the outer level considers a block of variables corresponding an instance , i.e., alpha_i
		double obj_temp;
		for(int order = 0; order < l; order++)
		{
			int id = perm[order];
			feature_node *xi;
			int yi = prob->y[id];

      double norm_id = norm_list[id];

      //solve the sub-problem (42) by Algorithm 6 and get the optimal z

			//compute v
			for(int iter_class = 0; iter_class < nr_class; iter_class++)
			{
				v[iter_class] = 0;
			}
			xi = prob->x[id];
			while(xi->index != -1)
			{
				int index = xi->index - 1;
				double value = xi->value;
				for(int iter_class = 0; iter_class < nr_class; iter_class++)
				{
					v[iter_class] += w[index*nr_class+iter_class] * value;
				}
				//compute K^i = norm(xi)^2
				//norm_list[id] += value * value;

				++xi;
			}

			//initialize z
			for(int iter_class = 0; iter_class < nr_class; iter_class++)
			{
				z[iter_class] = alpha[id*nr_class+iter_class];
			}
			
			//compute initial gradient
			//record G_max and G_min
			double G_max = -INF;
			int G_max_iter = -1;
			double G_min = INF;
			int G_min_iter = -1;
			for(int iter_class = 0; iter_class < nr_class; iter_class++)
			{
				G[iter_class] = log(z[iter_class]) + 1 - v[iter_class];
				if(G[iter_class] > G_max)
				{
					G_max = G[iter_class];
					G_max_iter = iter_class;
				}
				if(G[iter_class] < G_min)
				{
					G_min = G[iter_class];
					G_min_iter = iter_class;
				}
			}

      //the inner loop solves the sub-problem
			for(int inner_iter = 0; inner_iter < max_inner_iter; inner_iter++)
			//for(int inner_iter = 0; inner_iter < nr_class; inner_iter++)
			{
				//if(G_max == G_min)
				double dif = fabs((G_max - G_min) / G_min);
				//double dif = (G_max - G_min) / std::min(fabs(G_max),fabs(G_min));
				if( dif < g_eps )
				{
					break;
				}

        //calculate coefficients of (44) by using (47)
				double a = 2 * norm_id;
				double b = ( (z[G_max_iter]-alpha[id*nr_class+G_max_iter])-(z[G_min_iter]-alpha[id*nr_class+G_min_iter]) ) * norm_id;
				b += -v[G_max_iter] + v[G_min_iter];
				double c_1 = z[G_max_iter];
				double c_2 = z[G_min_iter];

				//solve (44) by Algorithm 4 and get the optimal z_1, z_2
				double z_1;
				double z_2;
				solve_sub_problem(a, b, c_1, c_2, &z_1, &z_2);

				//update z
				z[G_max_iter] = z_1;
				z[G_min_iter] = z_2;

				//update the gradient
				G[G_max_iter] = log(z_1) + 1 + norm_id * (z_1-alpha[id*nr_class+G_max_iter])- v[G_max_iter];
				G[G_min_iter] = log(z_2) + 1 + norm_id * (z_2-alpha[id*nr_class+G_min_iter])- v[G_min_iter];

				//find G_max and G_min
				G_max = -INF;
				G_min = INF;
				for(int iter_class = 0; iter_class < nr_class; iter_class++)
				{
				  if(G[iter_class] > G_max)
				  {
					  G_max = G[iter_class];
					  G_max_iter = iter_class;
				  }
				  if(G[iter_class] < G_min)
				  {
					  G_min = G[iter_class];
					  G_min_iter = iter_class;
				  }
				}
			}
			//end of solving the sub-problem

			//update w
			xi = prob->x[id];
			while(xi->index != -1)
			{
				int index = xi->index - 1;
				double value = xi->value;
				double temp = 0;
				for(int iter_class = 0; iter_class < nr_class; iter_class++)
				{
					temp = (z[iter_class]-alpha[id*nr_class+iter_class]);
           w[index*nr_class+iter_class] -= temp * value;
				}
				++xi;
			}
			//update alpha 
      for(int iter_class = 0; iter_class < nr_class; iter_class++)
			{
				alpha[id*nr_class+iter_class] = z[iter_class];
			}

			//obj_temp = compute_obj(w);
			//;

		}
		// one epoch of dual coordinate descent

		timer += clock() - start;

		primal = func->obj_primal(w);
		dual = func->obj_dual(alpha, w);

		func->grad(w, grad_w, &grad_norm);

		accuracy = func->testing(w);

		std::cout << iter << '\t' << timer << '\t' << primal << '\t' << dual << '\t' << accuracy << '\t' << grad_norm << std::endl;
		
    //compute objective
		obj[iter+1] = compute_obj(w);

		/*
		if(iter>5 && (fabs(obj[iter+1]-obj[iter])/obj[iter] <= eps) )
		{
			break;
		}
		 */
	}

	obj[iter+1] = -1;

	//delete [] norm_term;
	//delete [] probability;
	delete [] grad;
	delete [] grad_w;
	//delete [] obj;
  delete [] perm;
	delete [] output;
	delete [] v;
	delete [] z;
	delete [] G;


}

void Solver_ME_DUAL::solve_sub_problem(double a, double b, double c_1, double c_2, double *z_1, double *z_2)
{
	double s = c_1 + c_2;
  double z_middle = 0.5 * (c_2 - c_1);

  double c = c_1;
	int sign = 1;
	if(z_middle*a+b < 0)
	{
		c = c_2;
		sign = -1;
	}
	double z = c;
	if(s - z < 0.5 * s)
	{
		z *= 0.1;
	}
	//xi in the paper
	const double eta = 0.1;
	double innereps = 1e-3;

	double g_1;
	double g_2;
	double d;

	//a new modified Newton method for (18)
	int inner_iter;
	for(inner_iter = 0; inner_iter < max_inner_iter; inner_iter++)
	{
		//g'(z_t) = log(z_t/(s-z_t)) + a(z_t-c_t) + b_t
		g_1 = log(z/(s-z)) + a*(z-c) + sign*b;

		if(fabs(g_1) < innereps)
			break;

		//g''(z_t) = a + s/z_t/(s-z_t);
		g_2 = a + s/z/(s-z);
		d = - g_1 / g_2;

		//update z_t
		double z_temp = z + d;
		if(z_temp <= 0)
		{
			z *= eta;
		}
		else
		{
			z = z_temp;
		}
	}
	// end of Newton method
	
	//obtain z_1 and z_2
	if(inner_iter > 0)
	{
		if(sign == 1)
		{
			*z_1 = z;
			*z_2 = s - z;
		}
		else
		{
			*z_2 = z;
			*z_1 = s - z;
		}
	}
}

double Solver_ME_DUAL::compute_obj(double *w)
{
	double obj = 0;

	// ||W||_F
	for(int i = 0; i < w_size; i++)
	{

		obj += w[i] * w[i];
	}
	obj *= 0.5;

  // loss
  double loss_instance = 0;
	//double *output = Malloc(double, nr_class);
	double *output = new double[nr_class];
  double sum = 0;
  double normal_term = 0;
 
  for(int id = 0; id < l; id++)
	{
		for(int i = 0; i < nr_class; i++)
		{
			output[i] = 0;
		}

		feature_node *xi = prob->x[id];
		while(xi->index != -1)
		{
			int index = xi->index-1;
			double value = xi->value;
			for(int iter_class = 0; iter_class < nr_class; iter_class++)
			{
				output[iter_class] += w[index*nr_class+iter_class] * value;
				//output[iter_class] += w[iter_class*n+index] * value;
			}
      ++xi;
		}
 
    sum = 0;
		for(int i = 0; i < nr_class; i++)
		{
			sum += exp(output[i]);
		}
		sum = log(sum);
    
		int yi = prob->y[id];
		loss_instance -= output[yi]-sum;

	}

	loss_instance *= C;

	obj += loss_instance;

  delete [] output;

	return obj;
}

/*
double Solver_ME_DUAL::compute_dual_dif(double *w)
{
	double dual_dif = 0;
  
	double term_1 = 0;
	double term_2 = 0;
	double term_3 = 0;
	double term_4 = 0;

  double alpha_dif = 0;

	feature_node *xi = prob->x[id_current];
	int yi = prob->y[id_current];
	double norm_xi = 0;
	while(xi->index != -1)
	{
		norm_xi += xi->value * xi->value;
		++xi;
	}

	for(int iter_class = 0; iter_class < nr_class; iter_class++)
	{
		alpha_dif = alpha_new[iter_class] - alpha[id_current*nr_class+iter_class];

		term_1 = alpha_new[iter_class] * log(alpha_new[iter_class]);
		term_2 = alpha[id_current*nr_class+iter_class] * log(alpha[id_current*nr_class+iter_class]);
		
		term_3 = 0;
		xi = prob->x[id_current];
		while(xi->index != -1)
		{
			int index = xi->index - 1;
			double value = xi->value;
			term_3 += w[index*nr_class+iter_class] * value;
			++xi;
		}
		term_3 *= alpha_dif;

    term_4 = 0.5 * alpha_dif*alpha_dif * norm_xi;

    dual_dif += term_1 - term_2 + term_3 + term_4;
	}

	return dual_dif;
}
*/

//////////////////////////////////////////////
