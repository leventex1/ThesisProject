#pragma once
#include <sstream>
#include <math.h>
#include <memory>
#include <string>

#include "Core.h"
#include "../Math/Tensor2D.h"
#include "../Math/Operation.h"


namespace_start

enum OptimizerType
{
	None=-1,
	SGD, Adam,
};


class Optimizer
{
public:
	virtual ~Optimizer() { }

	virtual void Update(Tensor* params, Tensor* gradient, float learningRate) = 0;

	virtual std::string GetName() const = 0;
	virtual std::string ToString() const = 0;
	virtual void FromString(const std::string& fromString) = 0;
};


class SGDOptimizer : public Optimizer
{
public:
	SGDOptimizer(size_t numParams) { }
	SGDOptimizer(const std::string& fromString) { FromString(fromString); }
	virtual std::string GetName() const override { return "SGDOptimizer"; }
	virtual void Update(Tensor* params, Tensor* gradient, float learningRate) override
	{
		params->ElementWise(*gradient, [learningRate](float p, float g) -> float { return p - learningRate * g; });
	}
	virtual std::string ToString() const override 
	{ 
		std::stringstream ss;
		ss << GetName() << " ";
		return ss.str(); 
	}
	virtual void FromString(const std::string& fromString) override { }
	static std::string ClassName() { return "SGDOptimizer"; }
};


class AdamOptimizer : public Optimizer
{
public:
	AdamOptimizer(size_t numParams) : m_TrainingTimeStep(1), m_FirstMoments(numParams, 1), m_SecondMoments(numParams, 1) { }
	AdamOptimizer(const std::string& fromString) { FromString(fromString); }
	virtual std::string GetName() const override { return "AdamOptimizer"; }
	virtual void Update(Tensor* params, Tensor* gradient, float learningRate) override
	{
		const float b1 = 0.9f;
		const float b2 = 0.999f;
		const float ep = 1.0f / 100000000;
		size_t timeStep = m_TrainingTimeStep;

		m_FirstMoments.ElementWise(*gradient, [b1](float m, float g) -> float { return b1 * m + (1.0f - b1) * g; });
		m_SecondMoments.ElementWise(*gradient, [b2](float v, float g) -> float { return b2 * v + (1.0f - b2) * g * g; });

		Tensor2D correctedFirstMoments = Map(m_FirstMoments, [b1, timeStep](float m) -> float { return m / (1.0f - pow(b1, timeStep)); });
		Tensor2D correctedSecondMoments = Map(m_SecondMoments, [b2, timeStep](float v) -> float { return v / (1.0f - pow(b2, timeStep)); });

		Tensor2D correctedGradient = Map(correctedFirstMoments, [learningRate](float m) -> float { return learningRate * m; });
		correctedGradient.ElementWise(correctedSecondMoments, [ep](float v1, float v2) -> float { return v1 / (sqrt(v2) + ep); });

		params->Sub(correctedGradient);
		m_TrainingTimeStep++;
	}
	virtual std::string ToString() const override 
	{ 
		std::stringstream ss;
		ss << m_TrainingTimeStep << " " << m_FirstMoments.GetSize() << " ";
		for (size_t t = 0; t < m_FirstMoments.GetSize(); t++)
			ss << m_FirstMoments.GetData()[t] << " ";
		for (size_t t = 0; t < m_SecondMoments.GetSize(); t++)
			ss << m_SecondMoments.GetData()[t] << " ";
		return ss.str();
	}
	virtual void FromString(const std::string& fromString) override 
	{
		std::stringstream ss(fromString);
		size_t size;
		ss >> m_TrainingTimeStep >> size;
		m_FirstMoments = Tensor2D(size, 1);
		m_SecondMoments = Tensor2D(size, 1);
		for (size_t t = 0; t < size; t++)
			ss >> m_FirstMoments.GetData()[t];
		for (size_t t = 0; t < size; t++)
			ss >> m_SecondMoments.GetData()[t];
	};
	
	static std::string ClassName() { return "AdamOptimizer"; }
private:
	size_t m_TrainingTimeStep;
	Tensor2D m_FirstMoments;
	Tensor2D m_SecondMoments;
};


class OptimizerFactory
{
public:
	OptimizerFactory(OptimizerType type) : m_Type(type) { }
	OptimizerFactory() { }

	std::unique_ptr<Optimizer> Get(const std::string& fromString)
	{
		std::stringstream ss(fromString);
		std::string name;
		ss >> name;
		std::string remaining;
		std::getline(ss, remaining);

		if (name == SGDOptimizer::ClassName())		return std::make_unique<SGDOptimizer>(remaining);
		if (name == AdamOptimizer::ClassName())		return std::make_unique<AdamOptimizer>(remaining);

		throw std::exception("Unknown optimizer type!");
	}

	std::unique_ptr<Optimizer> Get(size_t numParams) const
	{
		switch (m_Type)
		{
		case OptimizerType::SGD:			return std::make_unique<SGDOptimizer>(numParams);
		case OptimizerType::Adam:			return std::make_unique<AdamOptimizer>(numParams);
		default:
			break;
		}

		throw std::exception("Unknown optimizer type!");
	}
private:
	OptimizerType m_Type = None;
};

namespace_end