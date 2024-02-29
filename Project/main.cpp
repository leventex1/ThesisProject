#include <iostream>
#include <memory>
#include <chrono>

#include <Mogi.h>
#include <MogiDataset.h>


class Timer {
public:
	Timer() {
		m_StartTime = std::chrono::system_clock::now();
	}
	~Timer() {
		std::chrono::time_point<std::chrono::system_clock>
			endTime = std::chrono::system_clock::now();

		double duration = std::chrono::duration<double>(endTime - m_StartTime).count();

		std::cout << "duration: " << duration << " [ms]" << std::endl;
	}
private:
	std::chrono::time_point<std::chrono::system_clock> m_StartTime;
};

int main(int argc, char* argv[])
{
	for (int i = 0; i < argc; i++)
	{
		std::cout << argv[i] << std::endl;
	}


	mogi::Model myModel("MNIST_conv_test_02.txt");
	// myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(28, 28, 1, 3, 3, 3, 1, mogi::RelU(), mogi::He(28 * 28)));
	// myModel.AddLayer(std::make_shared<mogi::MaxPoolingLayer>(28, 28, 3, 2, 2));
	// myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(14, 14, 3, 3, 3, 6, 1, mogi::RelU(), mogi::He(14 * 14)));
	// myModel.AddLayer(std::make_shared<mogi::MaxPoolingLayer>(14, 14, 6, 2, 2));
	// myModel.AddLayer(std::make_shared<mogi::ConvolutionalLayer>(7, 7, 6, 3, 3, 8, 1, mogi::RelU(), mogi::He(7 * 7)));
	// myModel.AddLayer(std::make_shared<mogi::FlattenLayer>(7, 7, 8, 392, 1, 1));
	// myModel.AddLayer(std::make_shared<mogi::DenseLayer>(392, 10, mogi::RelU(), mogi::He(392)));
	// myModel.AddLayer(std::make_shared<mogi::SoftmaxLayer>(10));


	myModel.Summarize();

	if (myModel.IsModelCorrect())
		std::cout << "Model is defined correctly." << std::endl;
	else
	{
		std::cout << "Model is not defined correctly." << std::endl;
		return -1;
	}


	mogi::dataset::MNISTDataset trainingDataset("Datasets/MNIST/train-images.idx3-ubyte", "Datasets/MNIST/train-labels.idx1-ubyte");
	mogi::dataset::MNISTDataset testingDataset("Datasets/MNIST/t10k-images.idx3-ubyte", "Datasets/MNIST/t10k-labels.idx1-ubyte");

	{
		float cost = 0.0f;
		float successCount = 0;
		for (size_t i = 0; i < testingDataset.GetEpochSize(); i++)
		{
			mogi::dataset::Sample testingSample = testingDataset.GetSample();
			testingDataset.Next();

			mogi::Tensor3D output = myModel.FeedForward(testingSample.Input);

			auto outputMaxPos = mogi::MaxPos(mogi::CreateWatcher(output, 0));
			auto labelMaxPos = mogi::MaxPos(mogi::CreateWatcher(testingSample.Label, 0));
			
			if (outputMaxPos.first == labelMaxPos.first)
				successCount++;

			mogi::CrossEntropyLoss CEL(testingSample.Label);
			cost += CEL.Cost(output);
		}
		std::cout << "Average cost before training: " << cost / testingDataset.GetEpochSize() << "\t Succes rate: " << successCount / testingDataset.GetEpochSize() << std::endl;
	}

	std::cout << "Training started..." << std::endl;
	for (size_t e = 0; e < 10; e++)
	{
		
		{
			Timer timer;
			for (size_t t = 0; t < trainingDataset.GetEpochSize(); t++)
			{
				mogi::dataset::Sample trainingSample = trainingDataset.GetSample();
				trainingDataset.Next();

				mogi::CrossEntropyLoss CEL(trainingSample.Label);
				myModel.BackPropagation(trainingSample.Input, CEL, 0.001f);
			}
		}

		trainingDataset.Shuffle();

		float cost = 0.0f;
		float successCount = 0;
		for (size_t i = 0; i < testingDataset.GetEpochSize(); i++)
		{
			mogi::dataset::Sample testingSample = testingDataset.GetSample();
			testingDataset.Next();

			mogi::Tensor3D output = myModel.FeedForward(testingSample.Input);

			auto outputMaxPos = mogi::MaxPos(mogi::CreateWatcher(output, 0));
			auto labelMaxPos = mogi::MaxPos(mogi::CreateWatcher(testingSample.Label, 0));

			if (outputMaxPos.first == labelMaxPos.first)
				successCount++;

			mogi::CrossEntropyLoss CEL(testingSample.Label);
			cost += CEL.Cost(output);
		}

		std::cout << "Epoch (" << e+1 << ") \t average cost: " << cost / testingDataset.GetEpochSize() << "\t Succes rate: " << successCount / testingDataset.GetEpochSize() << std::endl;
	}

	myModel.Save("MNIST_conv_test_01.txt");

	return 0;
}