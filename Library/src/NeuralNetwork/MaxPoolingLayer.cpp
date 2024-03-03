#include "MaxPoolingLayer.h"
#include <assert.h>
#include <sstream>


namespace_start

MaxPoolingLayer::MaxPoolingLayer(
	size_t inputHeight, size_t inputWidth, size_t inputDepth,
	size_t poolingHeight, size_t poolingWidth
) : m_InputHeight(inputHeight), m_InputWidth(inputWidth), m_InputDepth(inputDepth), m_PoolingWidth(poolingWidth), m_PoolingHeight(poolingHeight)
{
}

MaxPoolingLayer::MaxPoolingLayer(const std::string& fromString)
{
	FromString(fromString);
}

Tensor3D MaxPoolingLayer::FeedForward(const Tensor3D& inputs)
{
	assert(m_InputHeight == inputs.GetRows() &&
		m_InputWidth == inputs.GetCols() &&
		m_InputDepth == inputs.GetDepth()
		&& "Input shape not match!");

	LayerShape layerShape = GetLayerShape();

	Tensor3D output = Tensor3D(layerShape.OutputRows, layerShape.OutputCols, layerShape.OutputDepth);

	for (size_t d = 0; d < output.GetDepth(); d++)
	{
		Tensor2D slice = CreateWatcher((Tensor3D&)inputs, d);  // !

		for (size_t r = 0; r < output.GetRows(); r++)
		{
			for (size_t c = 0; c < output.GetCols(); c++)
			{
				Tensor2D window = CreateWatcher(slice, r * m_PoolingHeight, c * m_PoolingWidth, m_PoolingHeight, m_PoolingWidth, 0, 0);
				auto pos = MaxPos(window);

				output.SetAt(r, c, d, window.GetAt(pos.first, pos.second));
			}
		}
	}

	return output;
}

Tensor3D MaxPoolingLayer::BackPropagation(const Tensor3D& inputs, const CostFunction& costFucntion, float learningRate, size_t t)
{
	assert(m_InputHeight == inputs.GetRows() &&
		m_InputWidth == inputs.GetCols() &&
		m_InputDepth == inputs.GetDepth()
		&& "Input shape not match!");

	LayerShape layerShape = GetLayerShape();

	Tensor3D output = Tensor3D(layerShape.OutputRows, layerShape.OutputCols, layerShape.OutputDepth);
	
	for (size_t d = 0; d < output.GetDepth(); d++)
	{
		Tensor2D inputSlice = CreateWatcher((Tensor3D&)inputs, d);  // !

		for (size_t r = 0; r < output.GetRows(); r++)
		{
			for (size_t c = 0; c < output.GetCols(); c++)
			{
				Tensor2D window = CreateWatcher(inputSlice, r * m_PoolingHeight, c * m_PoolingWidth, m_PoolingHeight, m_PoolingWidth, 0, 0);
				auto pos = MaxPos(window);

				output.SetAt(r, c, d, window.GetAt(pos.first, pos.second));
			}
		}
	}

	Tensor3D costs = NextLayer ?
								NextLayer->BackPropagation(output, costFucntion, learningRate, t) :
								output;

	assert(layerShape.OutputRows == costs.GetRows() &&
		layerShape.OutputCols == costs.GetCols() &&
		layerShape.OutputDepth == costs.GetDepth()
		&& "Costs shape not match!");

	Tensor3D gradient = Tensor3D(layerShape.InputRows, layerShape.InputCols, layerShape.InputDepth);

	for (size_t d = 0; d < gradient.GetDepth(); d++)
	{
		Tensor2D inputSlice = CreateWatcher((Tensor3D&)inputs, d);  // !
		Tensor2D gradientSlice = CreateWatcher(gradient, d);

		for (size_t r = 0; r < costs.GetRows(); r++)
		{
			for (size_t c = 0; c < costs.GetCols(); c++)
			{
				Tensor2D inputWindow = CreateWatcher(inputSlice, r * m_PoolingHeight, c * m_PoolingWidth, m_PoolingHeight, m_PoolingWidth, 0, 0);
				Tensor2D gradientWindow = CreateWatcher(gradientSlice, r * m_PoolingHeight, c * m_PoolingWidth, m_PoolingHeight, m_PoolingWidth, 0, 0);
				auto pos = MaxPos(inputWindow);

				gradientWindow.SetAt(pos.first, pos.second, costs.GetAt(r, c, d));
			}
		}
	}

	return gradient;
}

LayerShape MaxPoolingLayer::GetLayerShape() const
{
	return
	{
		m_InputHeight, m_InputWidth, m_InputDepth,
		m_InputHeight / m_PoolingHeight, m_InputWidth / m_PoolingWidth, m_InputDepth
	};
}

std::string MaxPoolingLayer::ToString() const
{
	std::stringstream ss;

	ss << "[ " <<
		m_InputHeight << " " << m_InputWidth << " " << m_InputDepth << " " <<
		m_PoolingHeight << " " << m_PoolingWidth
		<< " ]";

	return ss.str();
}

void MaxPoolingLayer::FromString(const std::string& data)
{
	std::size_t numsStartPos = data.find(']');
	assert(data[0] == '[' && numsStartPos != std::string::npos && "Invalid hyperparameter format.");
	numsStartPos += 1;

	std::string hyperparams = data.substr(1, numsStartPos - 2);

	std::stringstream ss(hyperparams);

	try
	{
		ss >> m_InputHeight;
		ss >> m_InputWidth;
		ss >> m_InputDepth;
		ss >> m_PoolingHeight;
		ss >> m_PoolingWidth;
	}
	catch (...)
	{
		assert(false && "Invalid shape param!");
	}
}

std::string MaxPoolingLayer::ToDebugString() const
{
	std::stringstream ss;

	LayerShape shape = GetLayerShape();

	ss << "Input shape (" << shape.InputRows << ", " << shape.InputCols << ", " << shape.InputDepth << ")\n";
	ss << "Output shape (" << shape.OutputRows << ", " << shape.OutputCols << ", " << shape.OutputDepth << ")\n";

	return ss.str();
}

std::string MaxPoolingLayer::Summarize() const
{
	std::stringstream ss;

	LayerShape shape = GetLayerShape();

	ss << ClassName() << ":\t Input: (" <<
		shape.InputRows << ", " << shape.InputCols << ", " << shape.InputDepth << "), Output: (" <<
		shape.OutputRows << ", " << shape.OutputCols << ", " << shape.OutputDepth << "), " <<
		"Pooling: (" << m_PoolingHeight << ", " << m_PoolingWidth << ")" <<
		", # learnable parameters: " << 0;

	return ss.str();
}

namespace_end