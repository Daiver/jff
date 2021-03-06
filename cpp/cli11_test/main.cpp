#include <iostream>
#include "CLI11.hpp"

int main(int argc, char **argv)
{
    CLI::App app{"App description"};

    std::string filename = "default";
    app.add_option("-f,--file", filename, "A help string", true)->required();

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }
    std::cout << "END" << std::endl;
    std::cout << filename << std::endl;
    return 0;
}
