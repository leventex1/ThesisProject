#include <iostream>

#include <Mogi.h>
#include <MogiDataset.h>
#include "src/Classifier.h"

int main(int argc, char* argv[])
{
	for (int i = 0; i < argc; i++)
	{
		std::cout << argv[i] << std::endl;
	}

	mogi::Model myModel;
	myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(28, 28, 1, 3, 3, 3, 1, mogi::RelU(), mogi::Xavier(28 * 28, 28 * 28)));
	myModel.AddLayer(std::make_shared<mogi::MaxPoolingLayer>(28, 28, 3, 2, 2));
	myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(14, 14, 3, 3, 3, 6, 1, mogi::RelU(), mogi::Xavier(14 * 14, 14 * 14)));
	myModel.AddLayer(std::make_shared<mogi::MaxPoolingLayer>(14, 14, 6, 2, 2));
	myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(7, 7, 6, 3, 3, 8, 0, mogi::RelU(), mogi::Xavier(7 * 7, 5 * 5)));
	myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(5, 5, 8, 3, 3, 10, 0, mogi::RelU(), mogi::Xavier(5 * 5, 3 * 3)));
	myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(3, 3, 10, 3, 3, 12, 0, mogi::RelU(), mogi::Xavier(3 * 3, 1 * 1)));
	myModel.AddLayer(std::make_shared<mogi::FlattenLayer>(1, 1, 12, 12, 1, 1));
	myModel.AddLayer(std::make_shared<mogi::DenseLayer>(12, 10, mogi::RelU(), mogi::Xavier(12, 10)));
	myModel.AddLayer(std::make_shared<mogi::SoftmaxLayer>(10));

	mogi::dataset::MNISTDataset trainingDataset("Datasets/MNIST/train-images.idx3-ubyte", "Datasets/MNIST/train-labels.idx1-ubyte");
	mogi::dataset::MNISTDataset testingDataset("Datasets/MNIST/t10k-images.idx3-ubyte", "Datasets/MNIST/t10k-labels.idx1-ubyte");

	Classifier classifierTrainer(&myModel, &trainingDataset, &testingDataset, CostFunctionFactory(CostFunctionType::CrossEntropyLoss));

	classifierTrainer.Train(10, 0.01f, 0.001f);

	myModel.Save("MNIST_Conv_Classifier_Test_02.txt");

	return 0;
}