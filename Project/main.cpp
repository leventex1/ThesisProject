#include <iostream>

#include <Mogi.h>


int main(int argc, char* argv[])
{
	std::cout << "Hello World!" << std::endl;

	for (int i = 0; i < argc; i++)
	{
		std::cout << argv[i] << std::endl;
	}

	Test();


	return 0;
}