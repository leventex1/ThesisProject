#include "DenseLayer.h"
#include <assert.h>
#include <sstream>

#include "../Math/Operation.h"


namespace_start

DenseLayer::DenseLayer(size_t inputNodes, size_t outpuNodes, ActivationFunciton activationFunction)
	: m_ActivationFunction(activationFunction)
{
	m_Weights = Random2D(outpuNodes, inputNodes, -1.0f, 1.0f);
	m_Bias = Random2D(outpuNodes, 1, -1.0f, 1.0f);
}

DenseLayer::DenseLayer(const std::string& fromString)
{
	FromString(fromString);
}


std::vector<Tensor2D> DenseLayer::FeedForward(const std::vector<Tensor2D>& inputs) const
{
	assert(inputs.size() == 1 && "Dense layer's input is 1 Tensor2D!");

	const Tensor2D& input = inputs[0];

	Tensor2D sum = MatrixMult(m_Weights, input);
	sum.Add(m_Bias);
	sum.Map(m_ActivationFunction.Activation);

	return { sum };
}

std::vector<Tensor2D> DenseLayer::BackPropagation(const std::vector<Tensor2D>& inputs, const CostFunction& costFunction, float learningRate)
{
	assert(inputs.size() == 1 && "Dense layer's input is 1 Tensor2D!");
	assert(inputs[0].GetCols() == 1 && "Dense layer's input should contain 1 column!");
	assert(inputs[0].GetRows() == m_Weights.GetCols() && "Dense layer's input should contain as many rows as the weight's cols!");

	const Tensor2D& input = inputs[0];

	Tensor2D sum = MatrixMult(m_Weights, input);
	sum.Add(m_Bias);
	Tensor2D output = Map(sum, m_ActivationFunction.Activation);

	std::vector<Tensor2D> costs =	NextLayer ? 
									NextLayer->BackPropagation({ output }, costFunction, learningRate) : 
									costFunction.DiffCost({ output });

	assert(costs.size() == 1 && "Dense layer's cost is 1 Tensor2D");
	assert(costs[0].GetCols() == 1 && "Dense layer's cost should contain 1 column!");
	assert(costs[0].GetRows() == m_Weights.GetRows() && "Dense layer's cost should contain as many rows as the weight's rows!");

	const Tensor2D& cost = costs[0];
	sum.Map(m_ActivationFunction.DiffActivation);
	Tensor2D diffSum = Mult(cost, sum);

	Tensor2D gradWeights = MatrixMultRightTranspose(diffSum, input);
	Tensor2D& gradBiases = diffSum;
	Tensor2D gradCosts = MatrixMultLeftTranspose(m_Weights, diffSum);

	gradWeights.Map([learningRate](float v) -> float { return learningRate * v; });
	gradBiases.Map([learningRate](float v) -> float { return learningRate * v; });

	m_Weights.Sub(gradWeights);
	m_Bias.Sub(gradBiases);

	return { gradCosts };
}

LayerShape DenseLayer::GetLayerShape() const
{
	return 
	{
		m_Weights.GetCols(), 1, 1,
		m_Weights.GetRows(), 1, 1
	};
}

std::string DenseLayer::ToString() const
{
	std::stringstream ss;

	ss << "[ " << 
		m_Weights.GetCols() << " " << m_Weights.GetRows() << " " << 
		m_ActivationFunction.Name << " ( " << m_ActivationFunction.Params << " )" << 
	" ]";

	for (size_t t = 0; t < m_Weights.GetSize(); t++)
	{
		ss << " " << m_Weights.GetData()[t];
	}
	for (size_t t = 0; t < m_Bias.GetSize(); t++)
	{
		ss << " " << m_Bias.GetData()[t];
	}

	return ss.str();
}

void DenseLayer::FromString(const std::string& data)
{
	std::size_t numsStartPos = data.find(']');
	assert(data[0] == '[' && numsStartPos != std::string::npos && "Invalid hyperparameter format.");
	numsStartPos += 1;

	std::string hyperparams = data.substr(1, numsStartPos - 2);
	std::size_t acivationParamsStart = hyperparams.find('(');
	std::size_t acivationParamsEnd = hyperparams.find(')');
	assert(acivationParamsStart != std::string::npos && acivationParamsEnd != std::string::npos && "Invalid activation function params format.");

	std::stringstream ss(hyperparams);

	std::string inputNodesStr;
	std::string outputNodesStr;
	std::string activationFStr;
	std::string activationFParamsStr = hyperparams.substr(acivationParamsStart + 2, acivationParamsEnd - acivationParamsStart - 3);
	ss >> inputNodesStr;
	ss >> outputNodesStr;
	ss >> activationFStr;

	size_t inputNodes, outputNodes;

	try
	{
		inputNodes = std::stoi(inputNodesStr);
		outputNodes = std::stoi(outputNodesStr);
	}
	catch (...)
	{
		assert(false && "Invalid number.");
	}

	m_Weights = Tensor2D(outputNodes, inputNodes);
	m_Bias = Tensor2D(outputNodes, 1);
	m_ActivationFunction = GetActivationFunctionByName(activationFStr, activationFParamsStr);

	std::istringstream iss(data.substr(numsStartPos));

	for (size_t i = 0; i < m_Weights.GetSize(); i++)
	{
		iss >> m_Weights.GetData()[i];
	}
	for (size_t i = 0; i < m_Bias.GetSize(); i++)
	{
		iss >> m_Bias.GetData()[i];
	}
}

std::string DenseLayer::ToDebugString() const
{
	std::stringstream ss;
	ss << "Weights (" << m_Weights.GetRows() << ", " << m_Weights.GetCols() << ")\n";
	ss << m_Weights.ToString() << "\n";
	ss << "Bias (" << m_Bias.GetRows() << ", " << m_Bias.GetCols() << ")\n";
	ss << m_Bias.ToString() << "\n";
	return ss.str();
}

namespace_end