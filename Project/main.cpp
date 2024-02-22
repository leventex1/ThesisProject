#include <iostream>
#include <memory>

#include <Mogi.h>
#include <MogiDataset.h>


int main(int argc, char* argv[])
{
	for (int i = 0; i < argc; i++)
	{
		std::cout << argv[i] << std::endl;
	}

	mogi::dataset::XORDataset dataset;

	mogi::Model myModel;
	myModel.AddLayer(std::make_shared<mogi::DenseLayer>(2, 3, mogi::Sigmoid()));
	myModel.AddLayer(std::make_shared<mogi::DenseLayer>(3, 1, mogi::Sigmoid()));

	assert(dataset.CheckModelCompatibility(myModel) && "Model is not dataset compatible!");

	for(size_t epoch = 0; epoch < 1000; epoch++)
	{

		for (size_t t = 0; t < dataset.GetEpochSize(); t++)
		{
			mogi::dataset::Sample trainingSample = dataset.GetSample();
			dataset.Next();

			mogi::MeanSquareError MSE(trainingSample.Label);
			myModel.BackPropagation(trainingSample.Input, MSE, 1.0f);
		}

		float cost = 0.0f;
		for (size_t t = 0; t < dataset.GetEpochSize(); t++)
		{
			mogi::dataset::Sample trainingSample = dataset.GetSample();
			dataset.Next();

			mogi::MeanSquareError MSE(trainingSample.Label);
			std::vector<mogi::Tensor2D> output = myModel.FeedForward(trainingSample.Input);
			cost += MSE.Cost(output);
		}
		std::cout << "Epoch(" << epoch + 1 << ") cost: " << cost << std::endl;

		dataset.Shuffle();
	}


	return 0;
}