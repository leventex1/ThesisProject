#include "NearestUpsamplingLayer.h"
#include <assert.h>
#include <sstream>


namespace_start

NearestUpsamplingLayer::NearestUpsamplingLayer(
	size_t inputHeight, size_t inputWidth, size_t inputDepth,
	size_t poolingHeight, size_t poolingWidth
) : m_InputHeight(inputHeight), m_InputWidth(inputWidth), m_InputDepth(inputDepth), m_UpsamplingHeight(poolingWidth), m_UpsamplingWidth(poolingHeight)
{
}

NearestUpsamplingLayer::NearestUpsamplingLayer(const std::string& fromString)
{
	FromString(fromString);
}

Tensor3D NearestUpsamplingLayer::FeedForward(const Tensor3D& inputs)
{
	assert(m_InputHeight == inputs.GetRows() &&
		m_InputWidth == inputs.GetCols() &&
		m_InputDepth == inputs.GetDepth()
		&& "Input shape not match!");

	LayerShape layerShape = GetLayerShape();

	Tensor3D output = Tensor3D(layerShape.OutputRows, layerShape.OutputCols, layerShape.OutputDepth);

	for (size_t d = 0; d < inputs.GetDepth(); d++)
	{
		Tensor2D inputSlice = CreateWatcher((Tensor3D&)inputs, d);  // !
		Tensor2D outputSlice = CreateWatcher(output, d);

		for (size_t r = 0; r < inputs.GetRows(); r++)
		{
			for (size_t c = 0; c < inputs.GetCols(); c++)
			{
				float value = inputSlice.GetAt(r, c);
				Tensor2D outputWindow = CreateWatcher(outputSlice, r * m_UpsamplingHeight, c * m_UpsamplingWidth, m_UpsamplingHeight, m_UpsamplingWidth, 0, 0);
				for (size_t i = 0; i < outputWindow.GetSize(); i++)
				{
					size_t index = outputWindow.TraverseTo(i);
					outputWindow.GetData()[index] = value;
				}
			}
		}
	}

	return output;
}

Tensor3D NearestUpsamplingLayer::BackPropagation(const Tensor3D& inputs, const CostFunction& costFucntion, float learningRate)
{
	assert(m_InputHeight == inputs.GetRows() &&
		m_InputWidth == inputs.GetCols() &&
		m_InputDepth == inputs.GetDepth()
		&& "Input shape not match!");

	LayerShape layerShape = GetLayerShape();

	Tensor3D output = Tensor3D(layerShape.OutputRows, layerShape.OutputCols, layerShape.OutputDepth);

	for (size_t d = 0; d < inputs.GetDepth(); d++)
	{
		Tensor2D inputSlice = CreateWatcher((Tensor3D&)inputs, d);  // !
		Tensor2D outputSlice = CreateWatcher(output, d);

		for (size_t r = 0; r < inputs.GetRows(); r++)
		{
			for (size_t c = 0; c < inputs.GetCols(); c++)
			{
				float value = inputSlice.GetAt(r, c);
				Tensor2D outputWindow = CreateWatcher(outputSlice, r * m_UpsamplingHeight, c * m_UpsamplingWidth, m_UpsamplingHeight, m_UpsamplingWidth, 0, 0);
				for (size_t i = 0; i < outputWindow.GetSize(); i++)
				{
					size_t index = outputWindow.TraverseTo(i);
					outputWindow.GetData()[index] = value;
				}
			}
		}
	}

	Tensor3D costs = NextLayer ?
								NextLayer->BackPropagation(output, costFucntion, learningRate) :
								output;

	assert(layerShape.OutputRows == costs.GetRows() &&
		layerShape.OutputCols == costs.GetCols() &&
		layerShape.OutputDepth == costs.GetDepth()
		&& "Costs shape not match!");

	Tensor3D gradient = Tensor3D(layerShape.InputRows, layerShape.InputCols, layerShape.InputDepth);

	for (size_t d = 0; d < gradient.GetDepth(); d++)
	{
		Tensor2D gradientSlice = CreateWatcher(gradient, d);
		Tensor2D costSlice = CreateWatcher(costs, d);

		for (size_t r = 0; r < gradient.GetRows(); r++)
		{
			for (size_t c = 0; c < gradient.GetCols(); c++)
			{
				Tensor2D costWindow = CreateWatcher(costSlice, r * m_UpsamplingHeight, c * m_UpsamplingWidth, m_UpsamplingHeight, m_UpsamplingWidth, 0, 0);

				float value = 0.0f;
				for (size_t i = 0; i < costWindow.GetSize(); i++)
				{
					size_t index = costWindow.TraverseTo(i);
					value += costWindow.GetData()[index];
				}

				gradientSlice.SetAt(r, c, value);
			}
		}
	}

	return gradient;
}

LayerShape NearestUpsamplingLayer::GetLayerShape() const
{
	return
	{
		m_InputHeight, m_InputWidth, m_InputDepth,
		m_InputHeight * m_UpsamplingHeight, m_InputWidth * m_UpsamplingWidth, m_InputDepth
	};
}

std::string NearestUpsamplingLayer::ToString() const
{
	std::stringstream ss;

	ss << "[ " <<
		m_InputHeight << " " << m_InputWidth << " " << m_InputDepth << " " <<
		m_UpsamplingHeight << " " << m_UpsamplingWidth
		<< " ]";

	return ss.str();
}

void NearestUpsamplingLayer::FromString(const std::string& data)
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
		ss >> m_UpsamplingHeight;
		ss >> m_UpsamplingWidth;
	}
	catch (...)
	{
		assert(false && "Invalid shape param!");
	}
}

std::string NearestUpsamplingLayer::ToDebugString() const
{
	std::stringstream ss;

	LayerShape shape = GetLayerShape();

	ss << "Input shape (" << shape.InputRows << ", " << shape.InputCols << ", " << shape.InputDepth << ")\n";
	ss << "Output shape (" << shape.OutputRows << ", " << shape.OutputCols << ", " << shape.OutputDepth << ")\n";

	return ss.str();
}

std::string NearestUpsamplingLayer::Summarize() const
{
	std::stringstream ss;

	LayerShape shape = GetLayerShape();

	ss << ClassName() << ":\t Input: (" <<
		shape.InputRows << ", " << shape.InputCols << ", " << shape.InputDepth << "), Output: (" <<
		shape.OutputRows << ", " << shape.OutputCols << ", " << shape.OutputDepth << "), " <<
		"Pooling: (" << m_UpsamplingHeight << ", " << m_UpsamplingWidth << ")" <<
		", # learnable parameters: " << 0;

	return ss.str();
}

namespace_end