#include "App.hpp"
#include "Yolov10.hpp"
#include <iostream>

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <Image path>\n";
        return -1;
    }

    App app;
    app.run(argv[1]);
    return 0;
}
