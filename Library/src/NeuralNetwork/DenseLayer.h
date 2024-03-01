#pragma once
#include "Layer.h"
#include "ActivationF.h"
#include "Initializer.h"


namespace_start

class LIBRARY_API DenseLayer : public Layer
{
public:
	DenseLayer(size_t inputNodes, size_t outputNodes, ActivationFunciton activationFunction);
	DenseLayer(size_t inputNodes, size_t outputNodes, ActivationFunciton activationFunction, Initializer initializer);
	DenseLayer(const std::string& fromString);

	virtual Tensor3D FeedForward(const Tensor3D& inputs);
	virtual Tensor3D BackPropagation(const Tensor3D& inputs, const CostFunction& costFunction, float learningRate);

	virtual LayerShape GetLayerShape() const;

	virtual std::string GetName() const { return ClassName(); }
	virtual std::string ToString() const;
	virtual std::string ToDebugString() const;
	virtual std::string Summarize() const;

	virtual ActivationFunciton GetActivationFunction() const { return m_ActivationFunction; }
	virtual size_t GetLearnableParams() const { return m_Weights.GetSize() + m_Bias.GetSize(); };
	virtual std::string GetSepcialParams() const { return ""; };

	virtual void FromString(const std::string& data);

	static std::string ClassName() { return "DenseLayer"; }
private:
	Tensor2D m_Weights;
	Tensor2D m_Bias;
	ActivationFunciton m_ActivationFunction;
};

namespace_end
