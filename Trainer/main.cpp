#include <iostream>

#include <Mogi.h>
#include <MogiDataset.h>
#include "src/ClassificationTrainer.h"
#include "src/AutoencoderTrainer.h"
#include "src/Timer.h"


int main(int argc, char* argv[])
{
	for (int i = 0; i < argc; i++)
	{
		std::cout << argv[i] << std::endl;
	}


	mogi::Model myModel;
	myModel.AddLayer(std::make_shared<mogi::ReshapeLayer>(28, 28, 1, 784, 1, 1));
	myModel.AddLayer(std::make_shared<mogi::DenseLayer>(784, 64, mogi::RelU(), mogi::He(784)));
	myModel.AddLayer(std::make_shared<mogi::DenseLayer>(64, 784, mogi::Sigmoid(), mogi::Xavier(64, 784)));
	myModel.AddLayer(std::make_shared<mogi::ReshapeLayer>(784, 1, 1, 28, 28, 1));
	myModel.InitializeOptimizer(mogi::OptimizerFactory(mogi::OptimizerType::Adam));
	myModel.Summarize();

	//mogi::dataset::MNISTDataset testingDataset("Datasets/MNIST/t10k-images.idx3-ubyte", "Datasets/MNIST/t10k-labels.idx1-ubyte");
	//mogi::dataset::MNISTDataset trainingDataset("Datasets/MNIST/train-images.idx3-ubyte", "Datasets/MNIST/train-labels.idx1-ubyte");

	mogi::dataset::MNISTAutoEncoderDataset testingDataset("Datasets/MNIST/t10k-images.idx3-ubyte");
	mogi::dataset::MNISTAutoEncoderDataset trainingDataset("Datasets/MNIST/train-images.idx3-ubyte");
	
	AutoencoderTrainer trainer(&myModel, &trainingDataset, &testingDataset, CostFunctionFactory(MeanSuareError));

	float cost = trainer.Validate();
	std::cout << "Average cost before training: " << cost << std::endl;

	trainer.Train(1, 0.01f, 0.01f);	
	 
	//myModel.Save("MNIST_Dense_AutoEncoder_Test_06.txt");



	//mogi::Model myModel("MNIST_Conv_AutoEncoder_Test_05.txt");

	//std::cout << "AutoEncoder: " << std::endl;
	//myModel.Summarize();
	//

	//mogi::Model encoder;
	//mogi::Model decoder;

	//int decoderIndex = 2;

	//auto root = myModel.GetLayer(0);
	//auto bottleNeck = myModel.GetLayer(decoderIndex);

	//decoder.AddLayer(bottleNeck);
	//encoder.AddLayer(root);
	//encoder.GetLayer(decoderIndex - 1)->NextLayer = nullptr;

	////std::cout << "Encoder: " << std::endl;
	////encoder.Summarize();
	////std::cout << "Decoder: " << std::endl;
	////decoder.Summarize();

	////mogi::dataset::MNISTAutoEncoderDataset testingDataset("Datasets/MNIST/t10k-images.idx3-ubyte");

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