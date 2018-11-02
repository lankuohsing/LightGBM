#include <iostream>
#include <LightGBM/application.h>

int main(int argc, char** argv) {
  try {
	  char aa[2][200] = { "D:/Projects/github/LightGBM/windows/x64/Release/lightgbm.exe","config=D:/Projects/github/LightGBM/windows/x64/Release/train.conf" };
	  int ac = 2;
	  char *pp [2] = { aa[0],aa[1] };
    LightGBM::Application app(ac, pp);
    app.Run();
  }
  catch (const std::exception& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex.what() << std::endl;
    exit(-1);
  }
  catch (const std::string& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex << std::endl;
    exit(-1);
  }
  catch (...) {
    std::cerr << "Unknown Exceptions" << std::endl;
    exit(-1);
  }
}
