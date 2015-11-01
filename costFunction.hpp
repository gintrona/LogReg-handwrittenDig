//*************
//CostFunction 
//*************
using namespace std;

struct Params{
	Params(gsl_matrix * X , gsl_vector * y) : ptrToX(X), ptrToY(y){ };
	gsl_matrix * ptrToX;
	gsl_vector * ptrToY;
};

double costFunction(const gsl_vector *theta, void *params){

	Params* tmp = (Params*)params;

  	gsl_matrix *X = tmp->ptrToX;
  	gsl_vector *y = tmp->ptrToY;

	size_t num_examples = X->size1;
	size_t num_cols = X->size2;
		
  	gsl_vector *z = gsl_vector_alloc(num_examples);

	gsl_blas_dgemv(CblasNoTrans, 1.0, X, theta, 0, z);

	double acum=0;
	double acum_theta=0;

	for(size_t i=0 ; i<num_examples; i++){
		double h_theta = 1.0/(1.0 + gsl_sf_exp(-gsl_vector_get(z,i)));
		acum = acum + 
			(gsl_vector_get(y,i)==0? 0.0 : gsl_vector_get(y,i)*gsl_sf_log(h_theta)) +
			(gsl_vector_get(y,i)==1? 0.0 : (1.0-gsl_vector_get(y,i))*gsl_sf_log_1plusx(-h_theta))
			;
	}

  return -(1.0/num_examples)*acum;

}

/* The gradient of f, df = (df/dx, df/dy). */
void costFunctionGradient(const gsl_vector *theta, void *params, gsl_vector *df)
{

	Params* tmp 	= (Params*)params;
  	gsl_matrix *X 	= tmp->ptrToX;
  	gsl_vector *y 	= tmp->ptrToY;

	size_t num_examples = X->size1;
	size_t num_cols = X->size2;
	
  	gsl_vector *z = gsl_vector_alloc(num_examples);

  	gsl_blas_dgemv(CblasNoTrans, 1.0, X, theta, 0, z);

	for(size_t i = 0 ; i<num_cols;i++){//for each feature
		double acum =0;		
		for(size_t j = 0 ; j<num_examples;j++){
			double h_theta = 1.0/(1.0 + gsl_sf_exp(-gsl_vector_get(z,j)));
			acum = acum + gsl_matrix_get(X, j,i)*(h_theta - gsl_vector_get(y,j));
		}
		gsl_vector_set(df, i , acum/num_examples);
	}
}

/* Compute both f and df together. */
void costFunctionAll(const gsl_vector *x, void *params, double *f, gsl_vector *df)
{
  *f = costFunction(x, params); 
  costFunctionGradient(x, params, df);
}
