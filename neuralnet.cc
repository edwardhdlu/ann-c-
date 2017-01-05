#include "neuralnet.h"
#include <math.h>

using namespace std;

NeuralNet::NeuralNet(int input_size, int output_size, std::vector<int> hidden_sizes, Matrix inputs, Matrix outputs): 
input_size{input_size}, output_size{output_size}, hidden_sizes{hidden_sizes}, inputs{inputs}, outputs{outputs} {
	int prev_dim = inputs.getCols();
	for (unsigned int i = 0; i < hidden_sizes.size(); ++i) {
		Matrix weight_layer {prev_dim, hidden_sizes[i]};
		prev_dim = hidden_sizes[i];
		weight_layer.initNormal();
		weights.emplace_back(weight_layer);
		intermediates.emplace_back(Matrix {0, 0});
	}

	Matrix last_layer {prev_dim, outputs.getCols()};
	last_layer.initNormal();
	weights.emplace_back(last_layer);
	intermediates.emplace_back(Matrix {0, 0});
}

double NeuralNet::sigmoid(double n, bool deriv) {
	if (deriv) {
		return sigmoid(n) * (1 - sigmoid(n));
	}
	else {
		return 1 / (1 + exp(-n));
	}
}

double NeuralNet::relu(double n, bool deriv) {
	if (deriv) {
		return n > 0 ? 1 : 0;
	}
	else {
		return n > 0 ? n : 0;
	}
}

double NeuralNet::htan(double n, bool deriv) {
	if (deriv) {
		return 1 - pow(tanh(n), 2);
	}
	else {
		return tanh(n);
	}
}

Matrix NeuralNet::forwardProp() {
	Matrix result = inputs;
	for (unsigned int i = 0; i < weights.size(); ++i) {
		result = result * weights[i];
		result = activate(sigmoid, result);
		intermediates[i] = result;
	}

	return result;
}

void NeuralNet::backProp() {
	Matrix error = outputs - forwardProp(); 
	for (int i = intermediates.size() - 1; i >= 0; --i) {
		Matrix delta = error.unitMultiply(activate(sigmoid, intermediates[i], true));
		if (i == 0) {
			weights[i] = weights[i] + inputs.transpose() * delta;
		}
		else {
			weights[i] = weights[i] + intermediates[i - 1].transpose() * delta;
		}

		error = delta * weights[i].transpose();
	}
}

double square(double n) {
	return pow(n, 2.0);
}

double NeuralNet::loss() {
	Matrix result = forwardProp();
	Matrix delta = outputs - result;
	delta.apply(square);
	int size = delta.getRows() * delta.getCols();

	return delta.sum() / size;
}

Matrix NeuralNet::getWeights(int i) {
	return weights[i];
}

void NeuralNet::train(int epochs) {
	for (int i = 0; i < epochs; ++i) {
		backProp();
		if (i % 1000 == 0) {
			cout << "Error: " << loss() << " after " << i << " epochs" << endl;
		}
	}
}

Matrix NeuralNet::predict(Matrix input) {
	Matrix result = input;
	for (unsigned int i = 0; i < weights.size(); ++i) {
		result = result * weights[i];
		result = activate(sigmoid, result);
	}

	return result;
}