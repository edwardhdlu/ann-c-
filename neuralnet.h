#ifndef _NEURALNET_H_
#define _NEURALNET_H_
#include "matrix.h"

class NeuralNet {
public:
	// constructor
	NeuralNet(int input_size, int output_size, std::vector<int> hidden_sizes, Matrix inputs, Matrix outputs);

	// sigmoid and derivatives
	double sigmoid(double n);
	std::vector<double> sigmoid(std::vector<double> v);
	Matrix sigmoid(Matrix m);

	double sigmoidPrime(double n);
	std::vector<double> sigmoidPrime(std::vector<double> v);
	Matrix sigmoidPrime(Matrix m);

	// loss function (average sum of squares)
	double loss();

	// propogation
	Matrix forwardProp();
	void backProp();

	// ith weight matrix accessor
	Matrix getWeights(int i);

	// using the model
	void train(int epochs);
	Matrix predict(Matrix input);

private:
	const int input_size;
	const int output_size;
	const std::vector<int> hidden_sizes;

	std::vector<Matrix> intermediates;

	Matrix inputs;
	std::vector<Matrix> weights;
	std::vector<Matrix> biases;
	Matrix outputs;
};

#endif