#include "App.hpp"
#include "Yolov10.hpp"
#include <exception>
#include <iostream>

int main(int argc, char **argv)
{
    try {
        App app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
    }
    return 0;
}
