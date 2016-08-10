
#include "Function_SOFTMAX.h"

#include <math.h>


Function_SOFTMAX::Function_SOFTMAX(const problem *prob, const problem *prob_t, int nr_class, double C)
{
   this->prob = prob;
   this->prob_t = prob_t;
   this->C = C;

   this->nr_class = nr_class;
   this->l = prob->l;
   this->l_t = prob_t->l;
   this->n = prob->n;
   this->w_size = n * nr_class;

}

Function_SOFTMAX::~Function_SOFTMAX()
{

}

double Function_SOFTMAX::obj_primal(double *w)
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
   double sum = 0;
   double normal_term = 0;

   double *output = new double[nr_class];

   for(int id = 0; id < l; id++)
   {
      for(int iter_class = 0; iter_class < nr_class; iter_class++)
      {
         output[iter_class] = 0;
      }

      feature_node *xi = prob->x[id];
      while(xi->index != -1)
      {
         int index = xi->index - 1;
         double value = xi->value;
         for(int iter_class = 0; iter_class < nr_class; iter_class++)
         {
            output[iter_class] += w[index*nr_class+iter_class] * value;
         }
         ++xi;
      }

      sum = 0;
      for(int iter_class = 0; iter_class < nr_class; iter_class++)
      {
         sum += exp(output[iter_class]);
      }
      sum = log(sum);

      int yi = prob->y[id];
      loss_instance += (sum - output[yi]);
   }

   loss_instance *= C;

   obj += loss_instance;

   delete [] output;

   return obj;
}

double Function_SOFTMAX::obj_dual(double *alpha, double *w)
{
   double obj = 0;

   for(int i = 0; i < w_size; i++)
   {
      obj += w[i] * w[i];
   }
   obj *= 0.5;

   for(int id = 0; id < l; id++)
   {
      obj += (alpha[id] * log(alpha[id]));
   }

   return obj;

}

void Function_SOFTMAX::grad(double *w, double *grad, double *p_grad_norm)
{
   for(int i = 0; i < w_size; i++)
   {
      grad[i] = 0;
   }

   double *probability = new double[nr_class];

   for(int id = 0; id < l; id++)
   {
      feature_node *xi = prob->x[id];
      int yi = prob->y[id];


      for(int iter_class = 0; iter_class < nr_class; iter_class++)
      {
         probability[iter_class] = 0;
      }

      // softmax
      while(xi->index != -1)
      {
         int index = xi->index - 1;
         double value = xi->value;
         for(int iter_class = 0; iter_class < nr_class; iter_class++)
         {
            probability[iter_class] += w[index*nr_class+iter_class] * value;
         }
         ++xi;
      }

      double sum = 0;
      for(int iter_class = 0; iter_class < nr_class; iter_class++)
      {
         probability[iter_class] = exp(probability[iter_class]);
         sum += probability[iter_class];
      }
      for(int iter_class = 0; iter_class < nr_class; iter_class++)
      {
         probability[iter_class] /= sum;
      }

      // compute grad
      xi = prob->x[id];
      while(xi->index != -1)
      {
         int index = xi->index - 1;
         double value = xi->value;
         for(int iter_class = 0; iter_class < nr_class; iter_class++)
         {
            if(iter_class != yi)
            {
               grad[index*nr_class+iter_class] += w[index*nr_class+iter_class] - C * value * (0-probability[iter_class]);
            }
            grad[index*nr_class+yi] += w[index*nr_class+yi] - C * value * (1-probability[yi]);
         }
         ++xi;
      }
   }

   *p_grad_norm = 0;
   for(int i = 0; i < w_size; i++)
   {
      *p_grad_norm += grad[i] * grad[i];
   }
   *p_grad_norm = sqrt(*p_grad_norm);

   delete [] probability;
}

double Function_SOFTMAX::testing(double *w)
{
   int correct = 0;

   double *output = new double[nr_class];

   for(int id = 0; id < l_t; id++)
   {
      for(int iter_class = 0; iter_class < nr_class; iter_class++)
      {
         output[iter_class] = 0;
      }

      feature_node *xi = prob_t->x[id];
      int yi = prob_t->y[id];


      while(xi->index != -1)
      {
         int index = xi->index - 1;
         double value = xi->value;
         for(int iter_class = 0; iter_class < nr_class; iter_class++)
         {
            output[iter_class] += w[index*nr_class+iter_class] * value;
         }
         ++xi;
      }

      double max_value = -INF;
      int max_class = -1;
      for(int iter_class = 0; iter_class < nr_class; iter_class++)
      {
         if(output[iter_class] > max_value)
         {
            max_value = output[iter_class];
            max_class = iter_class;
         }
      }

      if(max_class == yi)
      {
         correct++;
      }
   }

   delete [] output;

   double accuracy = correct*1.0 / l_t;
   return accuracy;
}
