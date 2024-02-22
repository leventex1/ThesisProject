#pragma once
#include "Layer.h"
#include "ActivationF.h"


namespace_start

class LIBRARY_API DenseLayer : public Layer
{
public:
	DenseLayer(size_t inputNodes, size_t outputNodes, ActivationFunciton activationFunction);
	DenseLayer(const std::string& fromString);

	virtual std::vector<Tensor2D> FeedForward(const std::vector<Tensor2D>& inputs) const;
	virtual std::vector<Tensor2D> BackPropagation(const std::vector<Tensor2D>& inputs, const CostFunction& costFunction, float learningRate);

	virtual LayerShape GetLayerShape() const;

	virtual std::string GetName() const { return ClassName(); }
	virtual std::string ToString() const;
	virtual std::string ToDebugString() const;

	virtual void FromString(const std::string& data);

	static std::string ClassName() { return "DenseLayer"; }
private:
	Tensor2D m_Weights;
	Tensor2D m_Bias;
	ActivationFunciton m_ActivationFunction;
};

namespace_end
