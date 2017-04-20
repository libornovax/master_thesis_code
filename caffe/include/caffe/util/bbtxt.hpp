//
// Libor Novak
// 04/20/2017
//
// Functions for processing the BBTXT file format
//

#ifndef BBTXT_H
#define BBTXT_H

#include <opencv2/core/core.hpp>
#include <map>
#include "utils_bb.hpp"


/**
 * @brief Reads a BBTXT file into a map with bounding box lists indexed by filenames
 * @param path_bbtxt Path to the BBTXT file
 * @return
 */
std::map<std::string, std::vector<BB2D>> readBBTXTFile (const std::string &path_bbtxt);


#endif // BBTXT_H

