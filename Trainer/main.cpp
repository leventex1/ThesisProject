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

	mogi::Model myModel("MNIST_Conv_AutoEncoder_Adam_Test_01.txt");

	myModel.Summarize();

	//mogi::dataset::MNISTDataset testingDataset("Datasets/MNIST/t10k-images.idx3-ubyte", "Datasets/MNIST/t10k-labels.idx1-ubyte");
	//mogi::dataset::MNISTDataset trainingDataset("Datasets/MNIST/train-images.idx3-ubyte", "Datasets/MNIST/train-labels.idx1-ubyte");

	mogi::dataset::MNISTAutoEncoderDataset testingDataset("Datasets/MNIST/t10k-images.idx3-ubyte");
	mogi::dataset::MNISTAutoEncoderDataset trainingDataset("Datasets/MNIST/train-images.idx3-ubyte");
	
	AutoencoderTrainer trainer(&myModel, &trainingDataset, &testingDataset, CostFunctionFactory(MeanSuareError));

	float cost = trainer.Validate();
	std::cout << "Average cost before training: " << cost << std::endl;

	trainer.Train(1, 0.001f, 0.001f);	
	 
	myModel.Save("MNIST_Conv_AutoEncoder_Adam_Test_01-01.txt");


	//mogi::Model myModel("MNIST_Conv_AutoEncoder_Adam_Test_01-02.txt");

	//std::cout << "AutoEncoder: " << std::endl;
	//myModel.Summarize();
	//

	//mogi::Model encoder;
	//mogi::Model decoder;

	//int decoderIndex = 6;

	//auto root = myModel.GetLayer(0);
	//auto bottleNeck = myModel.GetLayer(decoderIndex);

	//decoder.AddLayer(bottleNeck);
	//encoder.AddLayer(root);
	//encoder.GetLayer(decoderIndex - 1)->NextLayer = nullptr;

	//std::cout << "Encoder: " << std::endl;
	//encoder.Summarize();
	//std::cout << "Decoder: " << std::endl;
	//decoder.Summarize();

	//mogi::dataset::MNISTAutoEncoderDataset testingDataset("Datasets/MNIST/t10k-images.idx3-ubyte");

	//mogi::dataset::Sample sample = testingDataset.GetSample();
	//testingDataset.Display();
	//std::cout << "------------------- |^ Original ^| -------------------" << std::endl;


	//mogi::Tensor3D latenCoord = encoder.FeedForward(sample.Input);

	//for (size_t i = 0; i < 10; i++)
	//{
	//	/*mogi::dataset::Sample sample = testingDataset.GetSample();
	//	testingDataset.Display();

	//	mogi::Tensor3D& input = sample.Input;*/

	//	mogi::Tensor3D latenOutput = latenCoord; //encoder.FeedForward(input);
	//	latenOutput.Add(mogi::Random3D(32, 1, 1, -2.0f, 2.0f));
	//	mogi::Tensor3D output = decoder.FeedForward(latenOutput);


	//	std::cout << "Laten coord: " << latenOutput.ToString() << std::endl;
	//	mogi::dataset::MNISTAutoEncoderDataset::Display(mogi::CreateWatcher(output, 0));
	//	std::cout << "-------------------" << std::endl;

	//	//testingDataset.Next();
	//}

	return 0;
}