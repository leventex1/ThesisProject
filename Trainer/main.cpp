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


	/*myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(28, 28, 1, 3, 3, 8, 1, mogi::RelU(), mogi::He(28 * 28)));
	myModel.AddLayer(std::make_shared<mogi::MaxPoolingLayer>(28, 28, 8, 2, 2));
	myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(14, 14, 8, 3, 3, 16, 0, mogi::RelU(), mogi::He(14 * 14)));
	myModel.AddLayer(std::make_shared<mogi::MaxPoolingLayer>(12, 12, 16, 2, 2));
	myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(6, 6, 16, 3, 3, 32, 0, mogi::RelU(), mogi::He(6 * 6)));
	myModel.AddLayer(std::make_shared<mogi::MaxPoolingLayer>(4, 4, 32, 2, 2));
	myModel.AddLayer(std::make_shared<mogi::ReshapeLayer>(2, 2, 32, 128, 1, 1));
	myModel.AddLayer(std::make_shared<mogi::DenseLayer>(128, 4, mogi::RelU(), mogi::He(128)));
	myModel.AddLayer(std::make_shared<mogi::DenseLayer>(4, 128, mogi::RelU(), mogi::He(10)));
	myModel.AddLayer(std::make_shared<mogi::ReshapeLayer>(128, 1, 1, 2, 2, 32));
	myModel.AddLayer(std::make_shared<mogi::NearestUpsamplingLayer>(2, 2, 32, 2, 2));
	myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(4, 4, 32, 3, 3, 16, 2, mogi::RelU(), mogi::He(4 * 4)));
	myModel.AddLayer(std::make_shared<mogi::NearestUpsamplingLayer>(6, 6, 16, 2, 2));
	myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(12, 12, 16, 3, 3, 8, 2, mogi::RelU(), mogi::He(12 * 12)));
	myModel.AddLayer(std::make_shared<mogi::NearestUpsamplingLayer>(14, 14, 8, 2, 2));
	myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(28, 28, 8, 3, 3, 1, 1, mogi::Sigmoid(), mogi::Xavier(28 * 28, 28 * 28)));*/


	mogi::Model myModel;
	myModel.AddLayer(std::make_shared<mogi::ReshapeLayer>(28, 28, 1, 784, 1, 1));
	myModel.AddLayer(std::make_shared<mogi::DenseLayer>(784, 64, mogi::RelU(0.001), mogi::He(784)));
	myModel.AddLayer(std::make_shared<mogi::DenseLayer>(64, 784, mogi::Sigmoid(), mogi::Xavier(64, 784)));
	myModel.AddLayer(std::make_shared<mogi::ReshapeLayer>(784, 1, 1, 28, 28, 1));
	myModel.Summarize();

	mogi::dataset::MNISTAutoEncoderDataset trainingDataset("Datasets/MNIST/train-images.idx3-ubyte");
	mogi::dataset::MNISTAutoEncoderDataset testingDataset("Datasets/MNIST/t10k-images.idx3-ubyte");

	AutoencoderTrainer trainer(&myModel, &trainingDataset, &testingDataset, CostFunctionFactory(MeanSuareError));

	float cost = trainer.Validate();
	std::cout << "Average cost before training: " << cost << std::endl;

	trainer.Train(1, 0.001f, 0.001f);
	 
	myModel.Save("MNIST_Dense_AutoEncoder_Test_05.txt");



	//mogi::Model myModel("MNIST_Conv_AutoEncoder_Test_05.txt");

	//std::cout << "AutoEncoder: " << std::endl;
	//myModel.Summarize();
	//

	//mogi::Model encoder;
	//mogi::Model decoder;

	//int decoderIndex = 8;

	//auto root = myModel.GetLayer(0);
	//auto bottleNeck = myModel.GetLayer(decoderIndex);

	//decoder.AddLayer(bottleNeck);
	//encoder.AddLayer(root);
	//encoder.GetLayer(decoderIndex - 1)->NextLayer = nullptr;

	////std::cout << "Encoder: " << std::endl;
	////encoder.Summarize();
	////std::cout << "Decoder: " << std::endl;
	////decoder.Summarize();

	//mogi::dataset::MNISTAutoEncoderDataset testingDataset("Datasets/MNIST/t10k-images.idx3-ubyte");

	//for (size_t i = 0; i < 10; i++)
	//{
	//	mogi::dataset::Sample sample = testingDataset.GetSample();
	//	testingDataset.Display();

	//	//mogi::Tensor3D input = mogi::Tensor3D(28 * 28, 1, 1, sample.Input.GetData());
	//	mogi::Tensor3D& input = sample.Input;

	//	mogi::Tensor3D latenOutput = encoder.FeedForward(input);
	//	mogi::Tensor3D output = decoder.FeedForward(latenOutput);

	//	//mogi::Tensor3D outputT = mogi::Tensor3D(28, 28, 1, output.GetData());
	//	mogi::Tensor3D& outputT = output;

	//	std::cout << "Laten coord: " << latenOutput.ToString() << std::endl;
	//	mogi::dataset::MNISTAutoEncoderDataset::Display(mogi::CreateWatcher(outputT, 0));
	//	std::cout << "-------------------" << std::endl;

	//	testingDataset.Next();
	//}

	return 0;
}