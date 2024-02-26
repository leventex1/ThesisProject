#pragma once
#include <functional>
#include <vector>
#include "Core.h"
#include <assert.h>
#include "../Math/Operation.h"
#include "../Math/Tensor3D.h"


namespace_start

struct LIBRARY_API CostFunction
{
	std::function<float(const Tensor3D& output)> Cost;
	std::function<Tensor3D(const Tensor3D& output)> DiffCost;
};

struct LIBRARY_API MeanSquareError : public CostFunction
{
	MeanSquareError(const Tensor3D& target)
	{
		Cost = [target](const Tensor3D& output) -> float {

			assert((output.GetDepth() == target.GetDepth() &&
					output.GetRows() == target.GetRows() &&
					output.GetCols() == target.GetCols()
				) && "Number of parameters in the output and target must be the same!");

			float cost = 0.0f;

			for (size_t t = 0; t < target.GetSize(); t++)
			{
				float diff = target.GetData()[t] - output.GetData()[t];
				cost += diff * diff;
			}

			return cost;
		};

		DiffCost = [target](const Tensor3D& output) -> Tensor3D {
			assert((output.GetDepth() == target.GetDepth() &&
				output.GetRows() == target.GetRows() &&
				output.GetCols() == target.GetCols()
				) && "Number of parameters in the output and target must be the same!");

			Tensor3D res = target;
			res.Sub(output);
			res.Map([](float v) -> float { return v * -2.0f; });

			return res;
		};
	}
};

struct LIBRARY_API CrossEntropyLoss : public CostFunction
{
	CrossEntropyLoss(const Tensor3D& target)
	{

		Cost = [target](const Tensor3D& output) -> float 
		{
			
			assert((output.GetDepth() == target.GetDepth() &&
				output.GetRows() == target.GetRows() &&
				output.GetCols() == target.GetCols()
				) && "Number of parameters in the output and target must be the same!");

			float cost = 0.0f;

			for (size_t t = 0; t < target.GetSize(); t++)
			{
				// TODO: log(0) -> Nan!
				float pred = output.GetData()[t];
				float tar = target.GetData()[t];
				cost += tar * log(pred);
			}

			return -1.0f * cost / target.GetSize();
		};

		DiffCost = [target](const Tensor3D& output) -> Tensor3D
		{
			assert((output.GetDepth() == target.GetDepth() &&
				output.GetRows() == target.GetRows() &&
				output.GetCols() == target.GetCols()
				) && "Number of parameters in the output and target must be the same!");

			Tensor3D res = output;
			res.Sub(target);
			return res;
		};
	}
};

namespace_end