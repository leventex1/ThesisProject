#include "DenseLayer.h"
#include <assert.h>
#include <sstream>

#include "../Math/Operation.h"


namespace_start

DenseLayer::DenseLayer(size_t inputNodes, size_t outputNodes, ActivationFunciton activationFunction)
	: m_ActivationFunction(activationFunction)
{
	m_Weights = Random2D(outputNodes, inputNodes, -1.0f, 1.0f);
	m_Bias = Random2D(outputNodes, 1, -1.0f, 1.0f);
}

DenseLayer::DenseLayer(size_t inputNodes, size_t outputNodes, ActivationFunciton activationFunction, Initializer initializer)
	: m_ActivationFunction(activationFunction)
{
	m_Weights = Tensor2D(outputNodes, inputNodes, initializer.Init);
	m_Bias = Tensor2D(outputNodes, 1, initializer.Init);
}

DenseLayer::DenseLayer(const std::string& fromString)
{
	FromString(fromString);
}

void DenseLayer::InitOptimizer(OptimizerFactory optimizerFactory)
{
	m_WeightsOptimizer = optimizerFactory.Get(m_Weights.GetSize());
	m_BiasOptimizer = optimizerFactory.Get(m_Bias.GetSize());
}

Tensor3D DenseLayer::FeedForward(const Tensor3D& inputs)
{
	assert(inputs.GetDepth() == 1 && "Dense layer's input is 1 Tensor2D!");

	Tensor2D input = SliceTensor(inputs, 0);

	Tensor2D sum = MatrixMult(m_Weights, input);
	sum.Add(m_Bias);
	sum.Map(m_ActivationFunction.Activation);

	return Tensor3D(sum.GetRows(), sum.GetCols(), 1, std::move(sum));
}

Tensor3D DenseLayer::BackPropagation(const Tensor3D& inputs, const CostFunction& costFunction, float learningRate, size_t t)
{
	if (!m_WeightsOptimizer || !m_BiasOptimizer)
		throw std::exception("No optimizer for training!");

	assert(inputs.GetDepth() == 1 && "Dense layer's input should contain a tensor with a depth of 1!");
	assert(inputs.GetCols() == 1 && "Dense layer's input should contain 1 column!");
	assert(inputs.GetRows() == m_Weights.GetCols() && "Dense layer's input should contain as many rows as the weight's cols!");

	LayerShape layerShape = GetLayerShape();

	Tensor2D input = SliceTensor(inputs, 0);

	Tensor2D sum = MatrixMult(m_Weights, input);
	sum.Add(m_Bias);
	Tensor3D output = Tensor3D(layerShape.OutputRows, layerShape.OutputCols, 1, std::move(Map(sum, m_ActivationFunction.Activation)));

	Tensor3D costs = NextLayer ? 
								NextLayer->BackPropagation(output, costFunction, learningRate, t) : 
								costFunction.DiffCost(output);

	assert(inputs.GetDepth() == 1 && "Dense layer's cost should contain a tensor with a depth of 1!");
	assert(costs.GetCols() == 1 && "Dense layer's cost should contain 1 column!");
	assert(costs.GetRows() == m_Weights.GetRows() && "Dense layer's cost should contain as many rows as the weight's rows!");

	Tensor2D cost = CreateWatcher(costs, 0);
	sum.Map(m_ActivationFunction.DiffActivation);
	Tensor2D diffSum = Mult(cost, sum);

	Tensor2D gradWeights = MatrixMultRightTranspose(diffSum, input);
	Tensor2D& gradBiases = diffSum;
	Tensor2D gradCosts = MatrixMultLeftTranspose(m_Weights, diffSum);

	m_WeightsOptimizer->Update(&m_Weights, &gradWeights, learningRate);
	m_BiasOptimizer->Update(&m_Bias, &gradBiases, learningRate);

	return Tensor3D(layerShape.InputRows, layerShape.InputCols, 1, std::move(gradCosts));
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

std::string DenseLayer::Summarize() const
{
	std::stringstream ss;

	LayerShape shape = GetLayerShape();

	ss << ClassName() << ":\t Input: (" <<
	shape.InputRows << ", " << shape.InputCols << ", " << shape.InputDepth << "), Output: (" <<
	shape.OutputRows << ", " << shape.OutputCols << ", " << shape.OutputDepth << "), " <<
	"Activation: " << m_ActivationFunction.Name << "(" << m_ActivationFunction.Params << "), " <<
	"# learnable parameters: " << m_Weights.GetSize() + m_Bias.GetSize();

	return ss.str();
}

//
//std::vector<Tensor3D> DenseLayer::BackPropagation(const std::vector<Tensor3D>& inputs, const CostFunction& costFunction, float learningRate)
//{
//	assert(inputs.size() >= 1 && "No input!");
//
//	LayerShape shape = GetLayerShape();
//
//	std::vector<Tensor3D> sums(inputs.size());
//	for (size_t t = 0; t < inputs.size(); t++)
//	{
//		sums.emplace_back(Tensor3D(shape.OutputRows, shape.OutputCols, shape.OutputDepth,
//			std::move(MatrixMult(m_Weights, CreateWatcher((Tensor3D&)inputs[t], 0)).Add(m_Bias))
//		));
//	}
//
//	std::vector<Tensor3D> outputs(inputs.size());
//	for (size_t t = 0; t < inputs.size(); t++)
//		outputs.emplace_back(Tensor3D(shape.OutputRows, shape.OutputCols, shape.OutputDepth, std::move(Map(sums[t], m_ActivationFunction.Activation))));
//
//	std::vector<Tensor3D> costs = NextLayer ? NextLayer->BackPropagation(outputs, costFunction, learningRate) : std::vector<Tensor3D>(inputs.size());
//	if (!NextLayer)
//		for (size_t t = 0; t < inputs.size(); t++)
//			costs.emplace_back(costFunction.DiffCost(outputs[t]));
//
//	for (size_t t = 0; t < inputs.size(); t++)
//	{
//		auto diffA = m_ActivationFunction.DiffActivation;
//		sums[t].ElementWise(costs[t], [diffA](float v1, float v2) -> float {
//			return v2 * diffA(v1);
//			});
//	}
//
//	Tensor2D gradWeights(m_Weights.GetRows(), m_Weights.GetCols());
//	Tensor2D gradBias(m_Bias.GetRows(), m_Bias.GetCols());
//	std::vector<Tensor3D> gradCosts(inputs.size());
//	for (size_t t = 0; t < inputs.size(); t++)
//	{
//		gradWeights.Add(MatrixMultRightTranspose(CreateWatcher(sums[t], 0), CreateWatcher((Tensor3D&)inputs[t], 0)));
//		gradWeights.Add(sums[t]);
//		gradCosts.emplace_back(MatrixMultLeftTranspose(m_Weights, CreateWatcher(sums[t], 0)));
//	}
//
//	size_t size = inputs.size();
//	gradWeights.Map([learningRate, size](float v) -> float { return learningRate * v / (float)size; });
//	gradBias.Map([learningRate, size](float v) -> float { return learningRate * v / (float)size; });
//
//	m_Weights.Sub(gradWeights);
//	m_Bias.Sub(gradBias);
//
//	return gradCosts;
//}


namespace_end