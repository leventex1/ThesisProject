#pragma once
#include <vector>
#include "Core.h"

#include <Mogi.h>


namespace_dataset_start

struct Sample
{
	std::vector<Tensor2D> Input;
	std::vector<Tensor2D> Label;
};

struct SampleShape
{
	size_t InputRows, InputCols, InputDepth;
	size_t LabelRows, LabelCols, LabelDepth;
};

class LIBRARY_API Dataset
{
public:
	virtual ~Dataset() { }

	virtual SampleShape GetSampleShape() const = 0;

	virtual Sample GetSample() const = 0;
	virtual size_t GetEpochSize() const = 0;

	virtual void Next() = 0;
	virtual void Shuffle() = 0;

	bool CheckModelCompatibility(const Model& model)
	{
		ModelShape modelShape = model.GetModelShape();
		SampleShape sampleShape = GetSampleShape();
		
		return (
			modelShape.InputRows == sampleShape.InputRows &&
			modelShape.InputCols == sampleShape.InputCols &&
			modelShape.InputDepth == sampleShape.InputDepth &&
			modelShape.OutputRows == sampleShape.LabelRows &&
			modelShape.OutputCols == sampleShape.LabelCols &&
			modelShape.OutputDepth == sampleShape.LabelDepth
		);
	}
};

namespace_dataset_end