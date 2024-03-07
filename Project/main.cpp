#include <iostream>
#include <chrono>

#include <Mogi.h>
#include <MogiDataset.h>

#include <opencv2/opencv.hpp>

class Timer {
public:
	Timer() {
		m_StartTime = std::chrono::system_clock::now();
	}
	double GetTime()
	{
		std::chrono::time_point<std::chrono::system_clock> endTime = std::chrono::system_clock::now();
		return std::chrono::duration<double>(endTime - m_StartTime).count();
	}
	~Timer()
	{
		std::chrono::time_point<std::chrono::system_clock> endTime = std::chrono::system_clock::now();
		double duration = std::chrono::duration<double>(endTime - m_StartTime).count();
		std::cout << "Duration: " << duration * 1000 << "ms" << std::endl;
	}
private:
	std::chrono::time_point<std::chrono::system_clock> m_StartTime;
};

int main(int argc, char* argv[])
{
	for (int i = 0; i < argc; i++)
	{
		std::cout << argv[i] << std::endl;
	}

	int index = 1;

	while (true)
	{
		std::string filePath = "images/image_" + std::to_string(index) + ".jpg";

		cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);

		if (image.empty())
		{
			std::cerr << "Could not open file at: " << filePath << std::endl;
			return -1;
		}

		cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);

		cv::imshow("Display window", image);

		index++;
		int key = cv::waitKey(0);

		if (key == 27)
		{
			break;
		}
	}

	return 0;
}