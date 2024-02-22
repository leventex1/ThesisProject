#pragma once
#include <functional>
#include <vector>
#include "Core.h"
#include <assert.h>
#include "../Math/Operation.h"
#include "../Math/Tensor2D.h"


namespace_start

struct LIBRARY_API CostFunction
{
	std::function<float(const std::vector<Tensor2D>& output)> Cost;
	std::function<std::vector<Tensor2D>(const std::vector<Tensor2D>& output)> DiffCost;
};

struct LIBRARY_API MeanSquareError : public CostFunction
{
	MeanSquareError(const std::vector<Tensor2D>& target)
	{
		Cost = [target](const std::vector<Tensor2D>& output) -> float {
			assert(output.size() > 0 && "No output!");
			assert(output.size() == target.size() && "Number of output and number of target must be the same!");

			float cost = 0.0f;

			for (size_t i = 0; i < output.size(); i++)
			{
				assert(target[i].GetSize() == output[i].GetSize() && "Number of parameters not match!");
				for (size_t t = 0; t < target[i].GetSize(); t++)
				{
					float diff = target[i].GetData()[t] - output[i].GetData()[t];
					cost += diff * diff;
				}
			}

			return cost;
		};

		DiffCost = [target](const std::vector<Tensor2D>& output) -> std::vector<Tensor2D> {
			assert(output.size() > 0 && "No output!");
			assert(output.size() == target.size() && "Number of output and number of target must be the same!");
			std::vector<Tensor2D> res;

			for (size_t i = 0; i < output.size(); i++)
			{
				assert(target[i].GetSize() == output[i].GetSize() && "Number of parameters not match!");

				res.push_back(Sub(target[i], output[i]));
				res[i].Map([](float v) -> float { return v * -2.0f; });
			}

			return res;
		};
	}
};

namespace_end