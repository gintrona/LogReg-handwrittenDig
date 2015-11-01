#include "oneVsAll.hpp"

using namespace std;

enum TYPE_OP {READ , COMPUTE, TEST};

void loadInput(string nameOfFile, gsl_matrix *TrainingX , gsl_vector *TrainingY );

int main(int argc,char *argv[]){

//Define the type of operation to perform
TYPE_OP OPERATION = COMPUTE;

size_t numExamples;
size_t numFeatures;
size_t input_layer_size;// 20x20 Input Images of Digits
size_t number_of_classes;
gsl_matrix * all_theta;
gsl_matrix *TrainingX;
gsl_vector *TrainingY;

if(OPERATION == READ || OPERATION == COMPUTE){

	//Define the size of the training set
	numExamples = 5000;
	numFeatures = 400; // 20x20 Input Images of Digits

	// Setup the parameters
	number_of_classes = 10;  // 10 labels, from 1 to 10   
		                 // ( "0" is mapped to label 10)

	//Each row of all_theta corresponds to a classifier; X and Y are the inputs.
	//All_theta is a matrix, each row corresponds to a classifier, each col is a feature
	all_theta = gsl_matrix_alloc(number_of_classes, numFeatures+1);

	TrainingX = gsl_matrix_alloc(numExamples, numFeatures+1);
	TrainingY = gsl_vector_alloc(numExamples);

	/*Loading Data = dataset contains handwritten digits.*/
	cout <<"Loading  Data ...\n"<<endl;

	loadInput("example.txt",TrainingX,TrainingY);
}

if(OPERATION == COMPUTE){
	
	one_vs_all(all_theta , TrainingX, TrainingY, number_of_classes);
	double accu;
	accu = calculateAccuracy(all_theta, TrainingX, TrainingY);
	cout<<"The accuracy is "<< accu<< "%."<<endl;
	//* Write result
	{
	FILE * f = fopen ("resu.dat", "wb");
	gsl_matrix_fwrite(f, all_theta);
	fclose (f);
	}
	gsl_matrix_free(all_theta);
}

// READ RESULT
if(OPERATION == READ){
     	FILE * f = fopen ("resu.dat", "rb");
     	gsl_matrix_fread (f, all_theta);
     	fclose (f);
	// Calculate the accuracy by comparing the input value and the
	// predicted value for each training item
	double accu;
	accu = calculateAccuracy(all_theta, TrainingX, TrainingY);
	cout<<"The accuracy is "<< accu<< "%."<<endl;
	gsl_matrix_free(all_theta);
}

if(OPERATION == TEST){

	//** define the size of the training set
	numExamples = 3;
	numFeatures = 3;

	TrainingX = gsl_matrix_alloc(numExamples, numFeatures+1);
	TrainingY = gsl_vector_alloc(numExamples);

	loadInput("exampleTest.txt", TrainingX, TrainingY);

	Params Training(TrainingX, TrainingY);

	gsl_vector * theta = gsl_vector_alloc(numFeatures+1);
	gsl_vector_set(theta,0,-2);
	gsl_vector_set(theta,1,-1);
	gsl_vector_set(theta,2,1);
	gsl_vector_set(theta,3,2);

	double resu = costFunction(theta, &Training);
	cout<<"Cost Function is "<<resu<< " (this value should be 4.6832)" <<endl; // -> 4.6832

	gsl_vector * df = gsl_vector_alloc(numFeatures+1);

	costFunctionAll(theta, &Training, &resu, df);//->//calls costFunctionGradient(theta, &Training , df);
	vector<float> expectedGrad =  {0.31722, 0.87232, 1.64812, 2.23787};

	cout<<"Cost Function by costFunctionAll is "<<resu<< " (this value should be 4.6832)" <<endl;
	for (int i=0 ; i<numFeatures+1 ; i++){
		cout<<gsl_vector_get(df,i)<< " ("<< expectedGrad.at(i)<<")"<< endl;
	}
	gsl_vector_free(theta);
}

//** release memory

gsl_matrix_free(TrainingX);
gsl_vector_free(TrainingY);

return 0;
}
