#include <assert.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "Model.h"
#include "DenseLayer.h"
#include "ConvolutionalLayer.h"
#include "SoftmaxLayer.h"
#include "ReshapeLayer.h"
#include "MaxPoolingLayer.h"


namespace_start

Model::Model(const std::string& filePath)
{
	Load(filePath);
}

void Model::AddLayer(const std::shared_ptr<Layer>& headLayer)
{
	if (!m_RootLayer)
	{
		m_RootLayer = headLayer;
		return;
	}

	std::shared_ptr<Layer> lastLayer = m_RootLayer;
	while (lastLayer->NextLayer) lastLayer = lastLayer->NextLayer;

	lastLayer->NextLayer = headLayer;
}

void Model::AddLayer(const std::string& layerName, const std::string& layerFromData)
{
	if (layerName == DenseLayer::ClassName()) { AddLayer(std::make_shared<DenseLayer>(layerFromData)); return; }
	if (layerName == ConvolutionalLayer::ClassName()) { AddLayer(std::make_shared<ConvolutionalLayer>(layerFromData)); return; }
	if (layerName == SoftmaxLayer::ClassName()) { AddLayer(std::make_shared<SoftmaxLayer>(layerFromData)); return; }
	if (layerName == ReshapeLayer::ClassName()) { AddLayer(std::make_shared<ReshapeLayer>(layerFromData)); return; }
	if (layerName == MaxPoolingLayer::ClassName()) { AddLayer(std::make_shared<MaxPoolingLayer>(layerFromData)); return; }

	assert(false && "Unknown layer name!");
}

Tensor3D Model::FeedForward(const Tensor3D& inputs) const
{
	assert(m_RootLayer != nullptr && "No layer available!");

	std::shared_ptr<Layer> layer = m_RootLayer;
	Tensor3D output = inputs;

	while (layer)
	{
		output = layer->FeedForward(output);
		layer = layer->NextLayer;
	}

	return output;
}

Tensor3D Model::BackPropagation(const Tensor3D& inputs, const CostFunction& costFunction, float learningRate)
{
	assert(m_RootLayer != nullptr && "No layer available!");

	return m_RootLayer->BackPropagation(inputs, costFunction, learningRate);
}

void Model::Save(const std::string& filePath) const
{
	assert(m_RootLayer != nullptr && "No layer available!");

	std::stringstream ss;

	std::shared_ptr<Layer> layer = m_RootLayer;
	while (layer)
	{
		ss << layer->GetName() << layer->ToString() << std::endl;
		layer = layer->NextLayer;
	}

	std::ofstream file(filePath);
	assert(file.is_open() && "Could not open file!");
	file << ss.str();
	file.close();
}

void Model::Load(const std::string& filePath)
{
	std::ifstream file(filePath);
	assert(file.is_open() && "File is cannot be opened.");

	std::string line;
	while (std::getline(file, line))
	{
		std::size_t endName = line.find('[');
		std::string layerName = line.substr(0, endName);
		std::string layerData = line.substr(endName);

		AddLayer(layerName, layerData);
	}
}

bool Model::IsModelCorrect(int* errorAt) const
{
	assert(m_RootLayer != nullptr && "No layer available!");

	std::shared_ptr<Layer> nextLayer = m_RootLayer->NextLayer;
	LayerShape layerShape = m_RootLayer->GetLayerShape();
	LayerShape nextLayerShape;

	int layerIndex = 0;
	while(nextLayer)
	{
		nextLayerShape = nextLayer->GetLayerShape();

		if (layerShape.OutputRows != nextLayerShape.InputRows ||
			layerShape.OutputCols != nextLayerShape.InputCols ||
			layerShape.OutputDepth != nextLayerShape.InputDepth)
		{
			*errorAt = layerIndex;
			return false;
		}

		layerShape = nextLayerShape;
		nextLayer = nextLayer->NextLayer;
		layerIndex++;
	}

	return true;
}

ModelShape Model::GetModelShape() const
{
	assert(m_RootLayer != nullptr && "No layer available!");

	LayerShape rootShape = m_RootLayer->GetLayerShape();

	std::shared_ptr<Layer> lastLayer = m_RootLayer;
	while (lastLayer->NextLayer) lastLayer = lastLayer->NextLayer;
	LayerShape headShape = lastLayer->GetLayerShape();

	return
	{
		rootShape.InputRows, rootShape.InputCols, rootShape.InputDepth,
		headShape.OutputRows, headShape.OutputCols, headShape.OutputDepth
	};
}

void Model::Summarize() const
{
	assert(m_RootLayer != nullptr && "No layer available!");

	std::shared_ptr<Layer> layer = m_RootLayer;

	while (layer)
	{
		LayerShape layerShape = layer->GetLayerShape();
		ActivationFunciton activatin = layer->GetActivationFunction();
		std::string special = layer->GetSepcialParams();

		std::cout << std::left;
		std::cout << std::setw(28) << (layer->GetName() + ": ");
		std::cout << std::setw(24) << ("Input: (" + std::to_string(layerShape.InputRows) + ", " + std::to_string(layerShape.InputCols) + ", " + std::to_string(layerShape.InputDepth) + ")" + ", ");
		std::cout << std::setw(24) << ("Output: (" + std::to_string(layerShape.OutputRows) + ", " + std::to_string(layerShape.OutputCols) + ", " + std::to_string(layerShape.OutputDepth) + ")" + ", ");
		std::cout << std::setw(32) << ("Activation: " + (activatin.Name.size() > 0 ? activatin.Name : "-") + " (" + (activatin.Params.size() > 0 ? activatin.Params : "-") + ")" + ", ");
		std::cout << std::setw(28) << ("# Learnable params: " + std::to_string(layer->GetLearnableParams()) + ", ");
		std::cout << std::setw(100) << ("Special: " + (special.size() > 0 ? special : "-"));
		std::cout << std::endl << std::right;


		layer = layer->NextLayer;
	}
}

const std::shared_ptr<Layer>& Model::GetLayer(size_t i)
{
	assert(m_RootLayer != nullptr && "No layer available!");
	if (!m_RootLayer)
		return m_RootLayer;

	std::shared_ptr<Layer> layer = m_RootLayer;
	for (size_t j = 0; j < i; j++)
	{
		if (!layer->NextLayer)
		{
			return layer;
		}
		layer = layer->NextLayer;
	}

	return layer;
}

namespace_end