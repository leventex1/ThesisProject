#include <iostream>
#include <iomanip>
#include "Timer.h"
#include "ClassificationTrainer.h"


ClassificationTrainer::ClassificationTrainer(
	mogi::Model* model, 
	mogi::dataset::Dataset* training, 
	mogi::dataset::Dataset* testing,
	CostFunctionFactory costFunctionFactory
)
	: Trainer(model, training, testing, costFunctionFactory)
{
}

float ClassificationTrainer::Validate(float* successRate) const
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

	if (successRate)
		*successRate = successCount / m_TestingDataset->GetEpochSize();
	return cost / m_TestingDataset->GetEpochSize();
}