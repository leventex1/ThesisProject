#include "AutoencoderTrainer.h"


AutoencoderTrainer::AutoencoderTrainer(
	mogi::Model* model,
	mogi::dataset::Dataset* training,
	mogi::dataset::Dataset* testing,
	CostFunctionFactory costFunctionFactory
) : Trainer(model, training, testing, costFunctionFactory)
{
}

float AutoencoderTrainer::Validate(float* successRate) const
{
	if(successRate)
		*successRate = -1.0f;

	float cost = 0.0f;
	for (size_t i = 0; i < m_TestingDataset->GetEpochSize(); i++)
	{
		mogi::dataset::Sample testingSample = m_TestingDataset->GetSample();
		m_TestingDataset->Next();

		mogi::Tensor3D output = m_Model->FeedForward(testingSample.Input);

		mogi::CostFunction loss = m_CostFunctionFactory.Build(testingSample.Label);
		cost += loss.Cost(output);
	}

	return cost / m_TestingDataset->GetEpochSize();
}