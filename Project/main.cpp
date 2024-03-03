#include <iostream>
#include <chrono>

#include <Mogi.h>
#include <MogiDataset.h>

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

	/*mogi::Tensor2D t1{  { 0.0f, 1.0f }, 
						{ 2.0f, 3.0f} };
	mogi::Tensor2D t2{  { 0.0f, 1.0f }, 
						{ 2.0f, 3.0f} };
	mogi::Tensor2D res = mogi::MatrixMult(t1, t2);
	std::cout << res.ToString() << std::endl;*/

	{
		Timer t;
		for (size_t t = 0; t < 100; t++)
		{
			mogi::Tensor2D t1(128, 784, 1.0f);
			mogi::Tensor2D t2(784, 1, -1.0f);
			mogi::Tensor2D res = mogi::MatrixMult(t1, t2);
		}
	}

	return 0;
}