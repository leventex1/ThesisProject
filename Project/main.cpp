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

	mogi::Tensor2D t1 = {
		{ 0.0f, 1.0f, 2.0f },
		{ 4.0f, 5.0f, 6.0f },
		{ 8.0f, 9.0f, 10.0f },
	};

	mogi::Tensor2D t2(3, 3, 1.0f);

	mogi::Tensor2D t3 = mogi::Convolution(t1, t2, 1, 1);

	std::cout << "t3: " << t3.ToString() << std::endl;

	return 0;

	mogi::dataset::MNISTDataset trainingDataset("Datasets/MNIST/train-images.idx3-ubyte", "Datasets/MNIST/train-labels.idx1-ubyte");
	mogi::dataset::MNISTDataset testingDataset("Datasets/MNIST/t10k-images.idx3-ubyte", "Datasets/MNIST/t10k-labels.idx1-ubyte");

	mogi::Model myModel;
	myModel.AddLayer(std::make_shared<mogi::DenseLayer>(784, 10, mogi::RelU(0.01), mogi::He(784) ));
	myModel.AddLayer(std::make_shared<mogi::DenseLayer>(10, 10, mogi::RelU(0.01), mogi::He(10) ));
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
		
		{
			Timer timer;
			for (size_t t = 0; t < trainingDataset.GetEpochSize(); t++)
			{
				mogi::dataset::Sample trainingSample = trainingDataset.GetSample();
				trainingDataset.Next();

				mogi::Tensor3D target = mogi::Tensor3D(10, 1, 1);
				target.SetAt(trainingSample.Label.GetAt(0, 0, 0), 0, 0, 1.0f);

				mogi::CrossEntropyLoss CEL(target);

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

	//myModel.Save("MNIST_model_01.txt");

	return 0;
}