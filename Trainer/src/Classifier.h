#pragma once
#include <Mogi.h>
#include <MogiDataset.h>
#include "CostFunctionFactory.h"


class Classifier
{
public:
	Classifier(
		mogi::Model* model,
		mogi::dataset::Dataset* training,
		mogi::dataset::Dataset* testing,
		CostFunctionFactory costFunctionFactory
	);

	void Train(
		size_t epochs,
		float startLearningRate,
		float endLearningRate
	);

	float Validate(float& succesRate) const;  // Returns the average cost.

private:
	mogi::Model* m_Model;
	mogi::dataset::Dataset* m_TrainingDataset;
	mogi::dataset::Dataset* m_TestingDataset;
	CostFunctionFactory m_CostFunctionFactory;
};
