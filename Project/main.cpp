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


	mogi::dataset::MNISTDataset trainingDataset("Datasets/MNIST/train-images.idx3-ubyte", "Datasets/MNIST/train-labels.idx1-ubyte");
	mogi::dataset::MNISTDataset testingDataset("Datasets/MNIST/t10k-images.idx3-ubyte", "Datasets/MNIST/t10k-labels.idx1-ubyte");

	mogi::Model myModel;
	myModel.AddLayer(std::make_shared<mogi::DenseLayer>(784, 20, mogi::Sigmoid()));
	myModel.AddLayer(std::make_shared<mogi::DenseLayer>(20, 10, mogi::Sigmoid()));
	myModel.AddLayer(std::make_shared<mogi::DenseLayer>(10, 10, mogi::Sigmoid()));
	myModel.AddLayer(std::make_shared<mogi::SoftmaxLayer>(10));

	if (myModel.IsModelCorrect())
		std::cout << "Model is defined correctly." << std::endl;
	else
	{
		std::cout << "Model is not defined correctly." << std::endl;
		return -1;
	}
	//assert(myModel.IsModelCorrect() && "Model definition not correct!");
	//assert(trainingDataset.IsModelCompatible(myModel) && "Model is not dataset compatible!");

	{
		float cost = 0.0f;
		float successCount = 0;
		for (size_t i = 0; i < testingDataset.GetEpochSize(); i++)
		{
			mogi::dataset::Sample testingSample = testingDataset.GetSample();
			testingDataset.Next();

			mogi::Tensor3D target = mogi::Tensor3D(10, 1, 1);
			target.SetAt(testingSample.Label.GetAt(0, 0, 0), 0, 0, 1.0f);

			mogi::Tensor3D output = myModel.FeedForward(testingSample.Input);

			float prediction = 0;
			int predictionIndex = 0;
			for (int i = 0; i < output.GetRows(); i++)
			{
				if (output.GetAt(i, 0, 0) > prediction)
				{
					prediction = output.GetAt(i, 0, 0);
					predictionIndex = i;
				}
			}
			if (predictionIndex == testingSample.Label.GetAt(0, 0, 0))
				successCount++;

			mogi::CrossEntropyLoss CEL(target);
			cost += CEL.Cost(output);
		}
		std::cout << "Average cost before training: " << cost / testingDataset.GetEpochSize() << "\t Succes rate: " << successCount / testingDataset.GetEpochSize() << std::endl;
	}

	std::cout << "Training started..." << std::endl;
	for (size_t e = 0; e < 10; e++)
	{
		for (size_t t = 0; t < trainingDataset.GetEpochSize(); t++)
		{
			mogi::dataset::Sample trainingSample = trainingDataset.GetSample();
			trainingDataset.Next();

			mogi::Tensor3D target = mogi::Tensor3D(10, 1, 1);
			target.SetAt(trainingSample.Label.GetAt(0, 0, 0), 0, 0, 1.0f);

			mogi::CrossEntropyLoss CEL(target);

			myModel.BackPropagation(trainingSample.Input, CEL, 1.0f);
		}
		trainingDataset.Shuffle();

		float cost = 0.0f;
		float successCount = 0;
		for (size_t i = 0; i < testingDataset.GetEpochSize(); i++)
		{
			mogi::dataset::Sample testingSample = testingDataset.GetSample();
			testingDataset.Next();

			mogi::Tensor3D target = mogi::Tensor3D(10, 1, 1);
			target.SetAt(testingSample.Label.GetAt(0, 0, 0), 0, 0, 1.0f);

			mogi::Tensor3D output = myModel.FeedForward(testingSample.Input);

			float prediction = 0;
			int predictionIndex = 0;
			for (int i = 0; i < output.GetRows(); i++)
			{
				if (output.GetAt(i, 0, 0) > prediction)
				{
					prediction = output.GetAt(i, 0, 0);
					predictionIndex = i;
				}
			}
			if (predictionIndex == testingSample.Label.GetAt(0, 0, 0))
				successCount++;

			mogi::CrossEntropyLoss CEL(target);
			cost += CEL.Cost(output);
		}

		std::cout << "Epoch (" << e+1 << ") \t average cost: " << cost / testingDataset.GetEpochSize() << "\t Succes rate: " << successCount / testingDataset.GetEpochSize() << std::endl;
	}

	//myModel.Save("testingMNIST.txt");

	return 0;
}