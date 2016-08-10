//////////////////////////////////////////////
//softmax
#include "Solver_SOFTMAX.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include <time.h>
#include <iostream>
//////////////////////////////////////////////

Solver_SOFTMAX::Solver_SOFTMAX(const problem *prob, int nr_class, double C, double eps, int max_iter, double eta)
{
	this->prob = prob;
	this->nr_class = nr_class;
	this->C = C;
	this->eps = eps;
	this->max_iter = max_iter;
  this->eta = eta;

	this->w_size = prob->n * nr_class;
	this->l = prob->l;
	this->n = prob->n;

}

Solver_SOFTMAX::~Solver_SOFTMAX()
{
}

void Solver_SOFTMAX::Solve(double *w, double *obj, Function_SOFTMAX *func)
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
	double *grad = new double[w_size];
	double *probability = new double[nr_class];
	double *norm_term = new double[nr_class];
  double sum = 0;
  //int id;

	double timer = 0;
	double primal;
	double dual;
	double grad_norm;
	double accuracy;
	double *grad_w = new double[w_size];

	for(int i = 0; i < w_size; i++)
		w[i] = 0;
		
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


		//stochastic gradient descent
		for(int order = 0; order < l; order++)
		{
			int id = perm[order];
			feature_node *xi = prob->x[id];
			int yi = prob->y[id];

			for(int i = 0; i < nr_class; i++)
			{
				probability[i] = 0;
			}

			//softmax
		  while(xi->index != -1)
		  {
			  int index = xi->index-1;
			  double value = xi->value;
			  for(int iter_class = 0; iter_class < nr_class; iter_class++)
			  {
				  probability[iter_class] += w[index*nr_class+iter_class] * value;
				  //probability[iter_class] += w[iter_class*n+index] * value;
			  }
        ++xi;
		  }
 
      sum = 0;
		  for(int i = 0; i < nr_class; i++)
		  {
				probability[i] = exp(probability[i]);
			  sum += probability[i];
		  }
      for(int i = 0; i < nr_class; i++)
		  {
				norm_term[i] = probability[i] / sum;
		  }

			//compute gradient
			xi = prob->x[id];
      while(xi->index != -1)
			{
				int index = xi->index-1;
				double value = xi->value;
				for(int iter_class = 0; iter_class < nr_class; iter_class++)
				{
					if(iter_class != yi)
					{
					  grad[index*nr_class+iter_class] = w[index*nr_class+iter_class] - C * value * (0-norm_term[iter_class]);
					  //grad[iter_class*n+index] = w[iter_class*n+index] - value * (0-norm_term[iter_class]);
					}
				}
				grad[index*nr_class+yi] = w[index*nr_class+yi] - C * value * (1-norm_term[yi]);
				//grad[yi*n+index] = w[yi*n+index] - value * (1-norm_term[yi]);

				++xi;
			}

      for(int i = 0; i < w_size; i++)
			{
				w[i] -= eta * grad[i];
			}

		}
		// one epoch of sgd

		timer += clock() - start;

		primal = func->obj_primal(w);
		accuracy = func->testing(w);

		func->grad(w, grad_w, &grad_norm);

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

	delete [] norm_term;
	delete [] probability;
	delete [] grad;
	delete [] grad_w;
	//delete [] obj;
  delete [] perm;

}

void Solver_SOFTMAX::solve_sub_problem()
{
}

double Solver_SOFTMAX::compute_obj(double *w)
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

//////////////////////////////////////////////
