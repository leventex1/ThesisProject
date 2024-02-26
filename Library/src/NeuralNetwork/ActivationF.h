#pragma once
#include <functional>
#include <string>
#include <assert.h>

#include "Core.h"


namespace_start

struct LIBRARY_API ActivationFunciton
{
	std::function<float(float v)> Activation;
	std::function<float(float v)> DiffActivation;

	std::string Name;
	std::string Params;
};

struct LIBRARY_API Sigmoid : public ActivationFunciton
{
	Sigmoid()
	{
		Activation = [](float v) -> float { return 1.0f / (1.0f + exp(-v)); };
		DiffActivation = [](float v) -> float { float sigm = 1.0f / (1.0f + exp(-v)); return sigm * (1.0f - sigm); };

		Name = "sigmoid";
		Params = "";
	}
};

struct LIBRARY_API RelU : public ActivationFunciton
{
	// Change alpha to get leakyRelU
	RelU(float alpha = 0.0f)
	{
		InitActivation(alpha);
		InitDiffActivation(alpha);
		Name = "RelU";
		Params = std::to_string(alpha);
	}

	RelU(const std::string& param)
	{
		try
		{
			float alpha = std::stof(param);
			InitActivation(alpha);
			InitDiffActivation(alpha);
			Name = "RelU";
			Params = std::to_string(alpha);
		}
		catch (...)
		{
			assert(false && "Invalid param!");
		}
	}

	void InitActivation(float alpha) { Activation = [alpha](float v) -> float { return v > 0 ? (1.0f + alpha) * v : alpha * v; }; }
	void InitDiffActivation(float alpha) { DiffActivation = [alpha](float v) -> float { return v > 0 ? (1.0f + alpha) : alpha; }; }
};

static ActivationFunciton GetActivationFunctionByName(const std::string& name, const std::string& params)
{
	if (name == "sigmoid") return Sigmoid();
	if (name == "RelU") return RelU(params);

	assert(false && "Unknown activation function name!");
	return ActivationFunciton();
}

namespace_end