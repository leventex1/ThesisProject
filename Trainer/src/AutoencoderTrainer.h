#pragma once
#include "Trainer.h"


class AutoencoderTrainer : public Trainer
{
public:
	AutoencoderTrainer(
		mogi::Model* model,
		mogi::dataset::Dataset* training,
		mogi::dataset::Dataset* testing,
		CostFunctionFactory costFunctionFactory = CostFunctionFactory(MeanSuareError)
	);

	virtual float Validate(float* successRate=nullptr) const;
};
