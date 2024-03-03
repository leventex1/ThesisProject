#pragma once
#include <math.h>
#include <memory>
#include <string>

#include "Core.h"
#include "../Math/Tensor2D.h"
#include "../Math/Operation.h"


namespace_start

enum OptimizerType
{
	SGD=0, Adam,
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
	virtual std::string GetName() const override { return "SGDOptimizer"; }
	virtual void Update(Tensor* params, Tensor* gradient, float learningRate) override
	{
		params->ElementWise(*gradient, [learningRate](float p, float g) -> float { return p - learningRate * g; });
	}
	virtual std::string ToString() const override { return ""; }
	virtual void FromString(const std::string& fromString) override { }
};


class AdamOptimizer : public Optimizer
{
public:
	AdamOptimizer(size_t numParams) : m_TrainingTimeStep(1), m_FirstMoments(numParams, 1), m_SecondMoments(numParams, 1) { }
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
	virtual std::string ToString() const override { return ""; }
	virtual void FromString(const std::string& fromString) override { };
private:
	size_t m_TrainingTimeStep;
	Tensor2D m_FirstMoments;
	Tensor2D m_SecondMoments;
};


class OptimizerFactory
{
public:
	OptimizerFactory(OptimizerType type) : m_Type(type) { }

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
	OptimizerType m_Type;
};


namespace_end