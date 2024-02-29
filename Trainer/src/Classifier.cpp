#include <iostream>
#include <iomanip>
#include "Timer.h"
#include "Classifier.h"


Classifier::Classifier(
	mogi::Model* model, 
	mogi::dataset::Dataset* training, 
	mogi::dataset::Dataset* testing,
	CostFunctionFactory costFunctionFactory
)
	: m_Model(model), m_TrainingDataset(training), m_TestingDataset(testing), m_CostFunctionFactory(costFunctionFactory)
{
	if (!m_Model->IsModelCorrect())
	{
		std::cout << "Model is not defined correctly!" << std::endl;
	}
}

void Classifier::Train(
	size_t epochs,
	float startLearningRate,
	float endLearningRate
)
{
	const size_t loadingBarTotal = 20;  // total number of steps for the loading bar.

	if (!m_TrainingDataset->IsModelCompatible(*m_Model))
	{
		std::cout << "The dataset is not compatible with the data!" << std::endl;
		return;
	}

	for (size_t e = 0; e < epochs; e++)
	{
		float learningRate = startLearningRate + ((float)e / (float)epochs) * (endLearningRate - startLearningRate);

		std::cout << "Epoch " << std::setw(7) << "(" + std::to_string(e + 1) + "/" + std::to_string(epochs) + ") ";
		std::cout << "[" << std::string(loadingBarTotal, ' ') << "] " << std::setw(4) << "0%";

		Timer timer;
		for (size_t t = 0; t < m_TrainingDataset->GetEpochSize(); t++)
		{
			mogi::dataset::Sample trainingSample = m_TrainingDataset->GetSample();
			m_TrainingDataset->Next();

			mogi::CostFunction loss = m_CostFunctionFactory.Build(trainingSample.Label);
			m_Model->BackPropagation(trainingSample.Input, loss, learningRate);

			if (t % (m_TrainingDataset->GetEpochSize() / 100) == 0 || t == m_TrainingDataset->GetEpochSize() - 1)
			{
				float status = std::min((float)t / (float)(m_TrainingDataset->GetEpochSize() - 1), 1.0f);
				size_t loadingStatus = status * loadingBarTotal;
				std::cout << "\r" << "Epoch " << std::setw(7) << "(" + std::to_string(e + 1) + "/" + std::to_string(epochs) + ") ";
				std::cout << "[" << std::string(loadingStatus, '=') << std::string(loadingBarTotal - loadingStatus, ' ') << "]";
				std::cout << std::setw(4) << (size_t)(status * 100) << "%";
			}
		}
		double duration = timer.GetTime();
		std::cout << " duration: " << duration << "[ms]";

		float successRate = 0.0f;
		float averageCost = Validate(successRate);

		std::cout << " Average cost: " << averageCost << " " << "Success rate: " << successRate << "%" << std::endl;

		m_TrainingDataset->Shuffle();
	}
}


float Classifier::Validate(float& successRate) const
{
	if (!m_TestingDataset->IsModelCompatible(*m_Model))
	{
		std::cout << "The dataset is not compatible with the data!" << std::endl;
		return 0.0f;
	}

	float cost = 0.0f;
	float successCount = 0;
	for (size_t i = 0; i < m_TestingDataset->GetEpochSize(); i++)
	{
		mogi::dataset::Sample testingSample = m_TestingDataset->GetSample();
		m_TestingDataset->Next();

		mogi::Tensor3D output = m_Model->FeedForward(testingSample.Input);

		auto outputMaxPos = mogi::MaxPos(mogi::CreateWatcher(output, 0));
		auto labelMaxPos = mogi::MaxPos(mogi::CreateWatcher(testingSample.Label, 0));

		if (outputMaxPos.first == labelMaxPos.first)
			successCount++;

		mogi::CostFunction loss = m_CostFunctionFactory.Build(testingSample.Label);
		cost += loss.Cost(output);
	}

	successRate = successCount / m_TestingDataset->GetEpochSize();
	return cost / m_TestingDataset->GetEpochSize();
}