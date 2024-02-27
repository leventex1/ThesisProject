#pragma once
#include <vector>
#include <memory>
#include <assert.h>
#include <string>
#include "Core.h"
#include "../Math/Tensor3D.h"
#include "CostF.h"


namespace_start

struct LayerShape
{
	size_t InputRows, InputCols, InputDepth;
	size_t OutputRows, OutputCols, OutputDepth;
};

class LIBRARY_API Layer
{
public:
	virtual ~Layer() { }

	virtual Tensor3D FeedForward(const Tensor3D& inputs) const = 0;

	/*
		Recursive backpropagation learning algorithm.
		First calculates the feedforward of the current layer.
		Than calls the nextLayer's backpropagation with the output as the next input.
		Than calculate the updated parameters respect to the returned costs.

		Returns the derivated cost respect to the inputs.

		TODO: implement bacch gradient descent.
	*/
	virtual Tensor3D BackPropagation(const Tensor3D& inputs, const CostFunction& costFunction, float learningRate) = 0;

	virtual LayerShape GetLayerShape() const = 0;

	virtual std::string GetName() const = 0;
	virtual std::string ToString() const = 0;
	virtual std::string ToDebugString() const { assert(false && "Debug string not implemented!"); return ""; }

	virtual void FromString(const std::string& data) = 0;

	std::shared_ptr<Layer> NextLayer = nullptr;
};

namespace_end
