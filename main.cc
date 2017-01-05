#include "matrix.h"
#include "neuralnet.h"
#include <iostream>

using namespace std;

int main() {
	Matrix inputs {4, 3, {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}}};
	Matrix outputs {4, 1, {{0}, {1}, {1}, {0}}};
	vector<int> hidden {4};

	NeuralNet nn {3, 1, hidden, inputs, outputs};
	nn.train(10000);

	Matrix test1 = nn.predict({1, 3, {{0, 0, 0}}});
	Matrix test2 = nn.predict({1, 3, {{1, 0, 0}}});
	Matrix test3 = nn.predict({1, 3, {{1, 1, 1}}});

	cout << test1;
	cout << test2;
	cout << test3;
}