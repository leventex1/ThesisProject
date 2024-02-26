#include "MNISTDataset.h"
#include <assert.h>
#include <random>
#include <iostream>


namespace_dataset_start

static int32_t ReadInt(std::ifstream& file) {
	int32_t result;
	file.read(reinterpret_cast<char*>(&result), sizeof(result));
	// Convert from big endian to little endian
	return (result >> 24) |
		((result << 8) & 0x00FF0000) |
		((result >> 8) & 0x0000FF00) |
		(result << 24);
}

MNISTDataset::MNISTDataset(const std::string& imagesFilePath, const std::string& labelsFilePath)
	: m_SampleIndex(0)
{
	size_t imagesCount = LoadImages(imagesFilePath);
	size_t labelsCount = LoadLabels(labelsFilePath);

	assert(imagesCount == labelsCount && "Count of images and labels not match!");

	m_EpochSize = imagesCount;
}

size_t MNISTDataset::LoadImages(const std::string& filePath)
{
	std::ifstream file(filePath, std::ios::binary);
	assert(file.is_open() && "Could not open MNIST images file!");

	int magicNumber = ReadInt(file);
	int imagesCount = ReadInt(file);
	int rows = ReadInt(file);
	int cols = ReadInt(file);

	assert(magicNumber == 2051 && "Invalid MNIST image file!");

	for (int t = 0; t < imagesCount; t++)
	{
		unsigned char data[28 * 28];
		file.read((char*)data, 28 * 28);
		m_Images.push_back(Tensor2D(28 * 28, 1));
		for (size_t i = 0; i < 28; i++)
		{
			for (size_t j = 0; j < 28; j++)
			{
				Tensor* tensor = &m_Images[t];
				tensor->SetAt(i * 28 + j, (float)data[i * 28 + j] / 255);
			}
		}
	}

	file.close();

	return imagesCount;
}

size_t MNISTDataset::LoadLabels(const std::string& filePath)
{
	std::ifstream file(filePath, std::ios::binary);
	assert(file.is_open() && "Could not open MNIST label file!");

	int magicNumber = ReadInt(file);
	int labelsCount = ReadInt(file);

	assert(magicNumber == 2049 && "Invalid MNIST label file!");

	unsigned char label = 0;
	for (int i = 0; i < labelsCount; i++)
	{
		file.read((char*)&label, sizeof(label));
		m_Labels.push_back(label);
	}

	file.close();

	return labelsCount;
}

SampleShape MNISTDataset::GetSampleShape() const
{
	return
	{
		28 * 28, 1, 1,
		1, 1, 1
	};
}

Sample MNISTDataset::GetSample() const
{
	return
	{
		Tensor3D(28 * 28, 1, 1, m_Images[m_SampleIndex].GetData()),
		{ { { (float)m_Labels[m_SampleIndex] } } }
	};
}

void MNISTDataset::Next()
{
	m_SampleIndex = (m_SampleIndex + 1) % m_EpochSize;
}

void MNISTDataset::Shuffle()
{
	std::random_device rd;

	for (size_t i = 0; i < 4; i++)
	{
		size_t swapIndex = rd() % m_EpochSize;
		std::swap(m_Images[i], m_Images[swapIndex]);
		std::swap(m_Labels[i], m_Labels[swapIndex]);
	}
}

void MNISTDataset::Display() const
{
	const std::string charSequence = " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@";

	Sample sample = GetSample();

	std::cout << "Label: " << sample.Label.GetAt(0, 0, 0) << std::endl;
	for (size_t i = 0; i < 28; i++)
	{
		for (size_t j = 0; j < 28; j++)
		{
			mogi::Tensor* input = &sample.Input;
			float value = input->GetAt(i * 28 + j);
 			char c = charSequence.at((int)(value * (charSequence.size() - 1)));
			std::cout << c << " ";
		}
		std::cout << "\n";
	}
}

namespace_dataset_end