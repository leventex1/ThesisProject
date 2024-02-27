#pragma once
#include "Layer.h"
#include "ActivationF.h"
#include "Initializer.h"


namespace_start

class LIBRARY_API ConvolutionalLayer : public Layer
{
public:
	ConvolutionalLayer();
	ConvolutionalLayer(const std::string& fromString);

	virtual Tensor3D FeedForward(const Tensor3D& inputs) const;
	virtual Tensor3D BackPropagation(const Tensor3D& inputs, const CostFunction& costFunction, float learningRate);

	virtual LayerShape GetLayerShape() const;

	virtual std::string GetName() const { return ClassName(); }
	virtual std::string ToString() const;
	virtual std::string ToDebugString() const;

	virtual void FromString(const std::string& data);

	static std::string ClassName() { return "ConvolutionalLayer"; }
private:
	ActivationFunciton m_ActivationFunction;
};

namespace_end
