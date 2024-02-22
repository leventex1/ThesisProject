#include <assert.h>
#include <sstream>
#include <fstream>
#include <iostream>

#include "Model.h"
#include "DenseLayer.h"


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
		m_HeadLayer = m_RootLayer;
		return;
	}

	m_HeadLayer->NextLayer = headLayer;
	m_HeadLayer = m_HeadLayer->NextLayer;
}

void Model::AddLayer(const std::string& layerName, const std::string& layerFromData)
{
	if (layerName == DenseLayer::ClassName()) AddLayer(std::make_shared<DenseLayer>(layerFromData)); return;

	assert(false && "Unknown layer name!");
}

std::vector<Tensor2D> Model::FeedForward(const std::vector<Tensor2D>& inputs) const
{
	assert(m_RootLayer != nullptr && "No layer available!");

	std::shared_ptr<Layer> layer = m_RootLayer;
	std::vector<Tensor2D> output = inputs;

	while (layer)
	{
		output = layer->FeedForward(output);
		layer = layer->NextLayer;
	}

	return output;
}

std::vector<Tensor2D> Model::BackPropagation(const std::vector<Tensor2D>& inputs, const CostFunction& costFunction, float learningRate)
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

bool Model::IsModelCorrect() const
{
	assert(m_RootLayer != nullptr && "No layer available!");

	std::shared_ptr<Layer> nextLayer = m_RootLayer->NextLayer;
	LayerShape layerShape = m_RootLayer->GetLayerShape();
	LayerShape nextLayerShape;


	while(nextLayer)
	{
		nextLayerShape = nextLayer->GetLayerShape();

		if (layerShape.OutputRows != nextLayerShape.InputRows ||
			layerShape.OutputCols != nextLayerShape.InputCols ||
			layerShape.OutputDepth != nextLayerShape.InputDepth)
			return false;

		layerShape = nextLayerShape;
	}

	return true;
}

ModelShape Model::GetModelShape() const
{
	assert(m_RootLayer != nullptr && "No layer available!");

	LayerShape rootShape = m_RootLayer->GetLayerShape();
	LayerShape headShape = m_HeadLayer->GetLayerShape();

	return
	{
		rootShape.InputRows, rootShape.InputCols, rootShape.InputDepth,
		headShape.OutputRows, headShape.OutputCols, headShape.OutputDepth
	};
}

namespace_end