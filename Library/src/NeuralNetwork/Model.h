#pragma once
#include <memory>
#include <vector>
#include <string>

#include "Core.h"
#include "Layer.h"


namespace_start

struct ModelShape
{
	size_t InputRows, InputCols, InputDepth;
	size_t OutputRows, OutputCols, OutputDepth;
};

class LIBRARY_API Model
{
public:
	Model() { }
	Model(const std::string& filePath);

	void AddLayer(const std::shared_ptr<Layer>& headLayer);
	void AddLayer(const std::string& layerName, const std::string& layerFromData);

	std::vector<Tensor2D> FeedForward(const std::vector<Tensor2D>& inputs) const;
	std::vector<Tensor2D> BackPropagation(const std::vector<Tensor2D>& inputs, const CostFunction& costFunction, float learningRate);

	void Save(const std::string& filePath) const;
	void Load(const std::string& filePath);

	bool IsModelCorrect() const;
	ModelShape GetModelShape() const;

private:
	std::shared_ptr<Layer> m_RootLayer = nullptr;
	std::shared_ptr<Layer> m_HeadLayer = nullptr;
};

namespace_end