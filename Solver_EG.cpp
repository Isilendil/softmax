//////////////////////////////////////////////
//exponentiated gradient
#include "Solver_EG.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include <time.h>
#include <algorithm>
#include <iostream>
//////////////////////////////////////////////

Solver_EG::Solver_EG(const problem *prob, int nr_class, double C, double eps, int max_iter, double eta_value, int max_trial, double initial)
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
	
	alpha_new = new double[nr_class];

  this->max_trial = max_trial;
	this->initial = initial;

  this->id_current = -1;

	eta = new double[l];
  for(int i = 0; i < l; i++)
	{
		eta[i] = eta_value;
	}

	norm_list = new double[l];
  for(int id = 0; id < l; id++)
	{
		norm_list[id] = 0;
	}
}

Solver_EG::~Solver_EG()
{
	delete [] alpha;
	delete [] eta;
	delete [] alpha_new;
	delete [] norm_list;
}

void Solver_EG::Solve(double *w, double *obj, Function_SOFTMAX *func)
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


		//online exponentiated gradient
		for(int order = 0; order < l; order++)
		{

			int id = perm[order];
			feature_node *xi;
			int yi = prob->y[id];
			id_current = id;

			eta[id] *= 0.97;

			// calculate v_y = w(alpha)^T f(x_i, y)
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
				++xi;
			}

			int LRATE_MAXCUTS = 8;
			int num_remaining_cuts = (iter==1)? 2 : LRATE_MAXCUTS;

      for(int iter_trial = 0; iter_trial < max_trial; iter_trial++)
			{
				xi = prob->x[id];
				
        //compute gradient
				for(int iter_class = 0; iter_class < nr_class; iter_class++)
				{
					grad[iter_class] = 1 + log(alpha[id*nr_class+iter_class]) + v[yi] - v[iter_class];
				}
				// w^T * ( f(x_i,y_i) - f(x_i,y) )
				/*
				while(xi->index != -1)
				{
					int index = xi->index - 1;
					double value = xi->value;
					for(int iter_class = 0; iter_class < nr_class; iter_class++)
					{
						if(iter_class != yi)
						{
							grad[iter_class] += (w[index*nr_class+yi]-w[index*nr_class+iter_class]) * value;
						}
					}
					++xi;
				}

				for(int iter_class = 0; iter_class < nr_class; iter_class++)
				{
					grad[iter_class] += 1 + log(alpha[id*nr_class+iter_class]);
				}
				*/

				//compute alpha_new
				double sum = 0;
        for(int iter_class = 0; iter_class < nr_class; iter_class++)
				{
					output[iter_class] = alpha[id*nr_class+iter_class] * exp(- eta[id] * grad[iter_class]);
					output[iter_class] = std::max(1e-100, output[iter_class]);
					sum += output[iter_class];
				}
				for(int iter_class = 0; iter_class < nr_class; iter_class++)
				{
					alpha_new[iter_class] = output[iter_class] / sum;
				}

				if(compute_dual_dif(v) < 0)
				{
					eta[id] *= 1.000;
					
					//update w
					xi = prob->x[id];
					while(xi->index != -1)
					{
						int index = xi->index - 1;
						double value = xi->value;
						double temp = 0;
						for(int iter_class = 0; iter_class < nr_class; iter_class++)
						{
							 temp = (alpha_new[iter_class]-alpha[id*nr_class+iter_class]);
               w[index*nr_class+iter_class] -= temp * value;
						}
						++xi;
					}

					//update alpha 
          for(int iter_class = 0; iter_class < nr_class; iter_class++)
					{
						alpha[id*nr_class+iter_class] = alpha_new[iter_class];
					}

					break;
				}
				else
				{
					if(num_remaining_cuts > 0)
					{
						eta[id] *= 0.5;
						num_remaining_cuts --;
					}
					else
					{
						break;
					}
				}
			}

		}
		// one epoch of eg

		timer += clock() - start;

		primal = func->obj_primal(w);
		dual = func->obj_dual(alpha, w);


		func->grad(w, grad_w, &grad_norm);

		accuracy = func->testing(w);

		std::cout << iter << '\t' << timer << '\t' << primal << '\t' << dual;
		std::cout << '\t' << accuracy << '\t' << grad_norm << std::endl;

    //compute objective
		obj[iter+1] = compute_obj(w);

		/*
		double dif = fabs((obj[iter+1]-obj[iter])/obj[iter]);
		if(iter>5 && dif <= eps)
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

}

void Solver_EG::solve_sub_problem()
{
}

double Solver_EG::compute_obj(double *w)
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

double Solver_EG::compute_dual_dif(double *v)
{
	double dual_dif = 0;
  
	double term_1 = 0;
	double term_2 = 0;
	double term_3 = 0;
	double term_4 = 0;

  double alpha_dif = 0;

	feature_node *xi = prob->x[id_current];
	int yi = prob->y[id_current];

  /*
	double norm_xi = 0;
	while(xi->index != -1)
	{
		norm_xi += xi->value * xi->value;
		++xi;
	}
	*/

	for(int iter_class = 0; iter_class < nr_class; iter_class++)
	{
		alpha_dif = alpha_new[iter_class] - alpha[id_current*nr_class+iter_class];

		term_1 = alpha_new[iter_class] * log(alpha_new[iter_class]);
		term_2 = alpha[id_current*nr_class+iter_class] * log(alpha[id_current*nr_class+iter_class]);

		/*
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
		 */
		term_3 = (alpha_new[iter_class] - alpha[id_current*nr_class+iter_class]) * v[iter_class];

    term_4 = 0.5 * alpha_dif*alpha_dif * norm_list[id_current];

    dual_dif += term_1 - term_2 + term_3 + term_4;
	}

	return dual_dif;
}

//////////////////////////////////////////////
