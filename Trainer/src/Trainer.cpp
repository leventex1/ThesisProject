#include <iostream>
#include <iomanip>
#include "Timer.h"
#include "Trainer.h"


Trainer::Trainer(
	mogi::Model* model,
	mogi::dataset::Dataset* training,
	mogi::dataset::Dataset* testing,
	CostFunctionFactory costFunctionFactory
)
	: m_Model(model), m_TrainingDataset(training), m_TestingDataset(testing), m_CostFunctionFactory(costFunctionFactory)
{
	int errorAt = -1;
	if (!m_Model->IsModelCorrect(&errorAt))
	{
		std::cout << "Model is not defined correctly! Error at layer: (" << errorAt << ")" << std::endl;
		throw -1;
	}
}

void Trainer::Train(
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
		for (size_t t = 0; t < m_TrainingDataset->GetEpochSize(); t += 1)
		{
			mogi::dataset::Sample trainingSample = m_TrainingDataset->GetSample();
			m_TrainingDataset->Next();

			mogi::CostFunction loss = m_CostFunctionFactory.Build(trainingSample.Label);
			m_Model->BackPropagation(trainingSample.Input, loss, learningRate, t);

			if ((t % std::max(size_t(1), (m_TrainingDataset->GetEpochSize() / 100)) == 0) || (t == (m_TrainingDataset->GetEpochSize() - 1)))
			{
				float status = std::min((float)t / (float)(m_TrainingDataset->GetEpochSize() - 1), 1.0f);
				size_t loadingStatus = status * loadingBarTotal;
				std::cout << "\r" << "Epoch " << std::setw(7) << "(" + std::to_string(e + 1) + "/" + std::to_string(epochs) + ") ";
				std::cout << "[" << std::string(loadingStatus, '=') << std::string(loadingBarTotal - loadingStatus, ' ') << "]";
				std::cout << std::setw(4) << (size_t)(status * 100) << "%";
			}
		}
		double duration = timer.GetTime();
		std::cout << " duration: " << duration << "[s]";

		float successRate = 0.0f;
		float averageCost = Validate(&successRate);

		std::cout << " Average cost: " << averageCost << " " << (successRate > -0.9f ? "Success rate: " + std::to_string(successRate) + "%" : "") << std::endl;

		m_TrainingDataset->Shuffle();
	}
}