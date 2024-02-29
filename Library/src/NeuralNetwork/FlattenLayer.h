#pragma once
#include "Layer.h"
#include "ActivationF.h"
#include "Initializer.h"


namespace_start

class LIBRARY_API FlattenLayer : public Layer
{
public:
	FlattenLayer(
		size_t inputHeight, size_t inputWidth, size_t inputDepth, 
		size_t outputHeight, size_t outputWidth, size_t outputDepth);
	FlattenLayer(const std::string& fromString);

	virtual Tensor3D FeedForward(const Tensor3D& inputs);
	virtual Tensor3D BackPropagation(const Tensor3D& inputs, const CostFunction& costFunction, float learningRate);

	virtual LayerShape GetLayerShape() const;

	virtual std::string GetName() const { return ClassName(); }
	virtual std::string ToString() const;
	virtual std::string ToDebugString() const;
	virtual std::string Summarize() const;

	virtual void FromString(const std::string& data);

	static std::string ClassName() { return "FlattenLayer"; }
private:
	size_t m_InputHeight, m_InputWidth, m_InputDepth;
	size_t m_OutputHeight, m_OutputWidth, m_OutputDepth;
};

namespace_end
