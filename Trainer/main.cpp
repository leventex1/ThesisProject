#include <iostream>

#include <Mogi.h>
#include <MogiDataset.h>
#include "src/ClassificationTrainer.h"
#include "src/AutoencoderTrainer.h"


int main(int argc, char* argv[])
{
	for (int i = 0; i < argc; i++)
	{
		std::cout << argv[i] << std::endl;
	}


	mogi::Model myModel;
	myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(28, 28, 1, 5, 5, 8, 2, mogi::RelU(0.01), mogi::He(28 * 28)));
	myModel.AddLayer(std::make_shared<mogi::MaxPoolingLayer>(28, 28, 8, 2, 2));
	myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(14, 14, 8, 5, 5, 10, 2, mogi::RelU(0.01), mogi::He(14 * 14)));
	myModel.AddLayer(std::make_shared<mogi::MaxPoolingLayer>(14, 14, 10, 2, 2));
	myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(7, 7, 10, 3, 3, 12, 0, mogi::RelU(0.01), mogi::He(7 * 7)));
	myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(5, 5, 12, 3, 3, 14, 0, mogi::RelU(0.01), mogi::He(5 * 5)));
	myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(3, 3, 14, 3, 3, 16, 0, mogi::RelU(0.01), mogi::He(3 * 3)));
	myModel.AddLayer(std::make_shared<mogi::ReshapeLayer>(1, 1, 16, 16, 1, 1));
	myModel.AddLayer(std::make_shared<mogi::DenseLayer>(16, 10, mogi::RelU(0.01), mogi::He(64)));
	myModel.AddLayer(std::make_shared<mogi::DenseLayer>(10, 16, mogi::RelU(0.01), mogi::He(32)));
	myModel.AddLayer(std::make_shared<mogi::ReshapeLayer>(16, 1, 1, 1, 1, 16));
	myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(1, 1, 16, 3, 3, 14, 2, mogi::RelU(0.01), mogi::He(1 * 1)));
	myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(3, 3, 14, 3, 3, 12, 2, mogi::RelU(0.01), mogi::He(3 * 3)));
	myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(5, 5, 12, 3, 3, 10, 2, mogi::RelU(0.01), mogi::He(5 * 5)));
	myModel.AddLayer(std::make_shared<mogi::NearestUpsamplingLayer>(7, 7, 10, 2, 2));
	myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(14, 14, 10, 5, 5, 8, 2, mogi::RelU(0.01), mogi::He(14 * 14)));
	myModel.AddLayer(std::make_shared<mogi::NearestUpsamplingLayer>(14, 14, 8, 2, 2));
	myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(28, 28, 8, 5, 5, 1, 2, mogi::Sigmoid(), mogi::He(28 * 28)));
	myModel.Summarize();

	mogi::dataset::MNISTAutoEncoderDataset trainingDataset("Datasets/MNIST/train-images.idx3-ubyte");
	mogi::dataset::MNISTAutoEncoderDataset testingDataset("Datasets/MNIST/t10k-images.idx3-ubyte");

	AutoencoderTrainer trainer(&myModel, &trainingDataset, &testingDataset);

	float cost = trainer.Validate();
	std::cout << "Average cost before training: " << cost << std::endl;

	trainer.Train(3, 0.01f, 0.001f);
	 
	myModel.Save("MNIST_Conv_AutoEncoder_Test_01.txt");


	/*mogi::dataset::MNISTAutoEncoderDataset testingDataset("Datasets/MNIST/t10k-images.idx3-ubyte");

	mogi::Model myModel("MNIST_Dense_AutoEncoder_Test_02.txt");

	mogi::Model encoder;
	mogi::Model decoder;

	auto root = myModel.GetLayer(0);
	auto bottleNeck = myModel.GetLayer(2);

	decoder.AddLayer(bottleNeck);
	encoder.AddLayer(root);
	encoder.GetLayer(1)->NextLayer = nullptr;

	encoder.Summarize();
	decoder.Summarize();

	for (size_t i = 0; i < 10; i++)
	{
		mogi::dataset::Sample sample = testingDataset.GetSample();
		testingDataset.Display();

		mogi::Tensor3D latenOutput = encoder.FeedForward(sample.Input);
		mogi::Tensor3D output = decoder.FeedForward(latenOutput);

		std::cout << "Laten coord: " << latenOutput.ToString() << std::endl;
		mogi::dataset::MNISTAutoEncoderDataset::Display(mogi::CreateWatcher(output, 0));
		std::cout << "-------------------" << std::endl;

		testingDataset.Next();
	}*/

	return 0;
}