#include <caffe/caffe.hpp>
#include <boost/algorithm/string.hpp>

#include "caffe/util/bbtxt.hpp"


std::map<std::string, std::vector<BB2D>> readBBTXTFile (const std::string &path_bbtxt)
{
    std::ifstream infile(path_bbtxt.c_str(), std::ios::in);
    CHECK(infile.is_open()) << "BBTXT file '" << path_bbtxt << "' could not be opened!";

    std::string line;
    std::vector<std::string> data;

    std::map<std::string, std::vector<BB2D>> bbs_out;

    // Read the whole file and create entries in the bbs_out
    while (std::getline(infile, line))
    {
        // Split the line - entries separated by space [filename label confidence xmin ymin xmax ymax]
        boost::split(data, line, boost::is_any_of(" "));
        CHECK_EQ(data.size(), 7) << "Line '" << line << "' corrupted!";

        // The [] operator on the map creates a new entry if the key is missing, otherwise returns
        // the existing entry - that is exactly what we need
        bbs_out[data[0]].emplace_back(data[0], std::stod(data[1]), std::stod(data[2]), std::stod(data[3]),
                                      std::stod(data[4]), std::stod(data[5]), std::stod(data[6]));
    }

    infile.close();

    return bbs_out;
}
