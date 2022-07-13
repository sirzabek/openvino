// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fileutils.hpp"

void ArkFile::get_file_info(const char* fileName,
                            uint32_t numArrayToFindSize,
                            uint32_t* ptrNumArrays,
                            uint32_t* ptrNumMemoryBytes) {
    uint32_t numArrays = 0;
    uint32_t numMemoryBytes = 0;

    std::ifstream in_file(fileName, std::ios::binary);
    if (in_file.good()) {
        while (!in_file.eof()) {
            std::string line;
            uint32_t numRows = 0u, numCols = 0u, num_bytes = 0u;
            std::getline(in_file, line, '\0');  // read variable length name followed by space and NUL
            std::getline(in_file, line, '\4');  // read "BFM" followed by space and control-D
            if (line.compare("BFM ") != 0) {
                break;
            }
            in_file.read(reinterpret_cast<char*>(&numRows), sizeof(uint32_t));  // read number of rows
            std::getline(in_file, line, '\4');                                  // read control-D
            in_file.read(reinterpret_cast<char*>(&numCols), sizeof(uint32_t));  // read number of columns
            num_bytes = numRows * numCols * sizeof(float);
            in_file.seekg(num_bytes, in_file.cur);  // read data

            if (numArrays == numArrayToFindSize) {
                numMemoryBytes += num_bytes;
            }
            numArrays++;
        }
        in_file.close();
    } else {
        throw std::runtime_error(std::string("Failed to open %s for reading in get_file_info()!\n") + fileName);
    }

    if (ptrNumArrays != NULL)
        *ptrNumArrays = numArrays;
    if (ptrNumMemoryBytes != NULL)
        *ptrNumMemoryBytes = numMemoryBytes;
}

void ArkFile::load_file(const char* fileName,
                        uint32_t arrayIndex,
                        std::string& ptrName,
                        std::vector<uint8_t>& memory,
                        uint32_t* ptrNumRows,
                        uint32_t* ptrNumColumns,
                        uint32_t* ptrNumBytesPerElement) {
    std::ifstream in_file(fileName, std::ios::binary);
    if (in_file.good()) {
        uint32_t i = 0;
        while (i < arrayIndex) {
            std::string line;
            uint32_t numRows = 0u, numCols = 0u;
            std::getline(in_file, line, '\0');  // read variable length name followed by space and NUL
            std::getline(in_file, line, '\4');  // read "BFM" followed by space and control-D
            if (line.compare("BFM ") != 0) {
                break;
            }
            in_file.read(reinterpret_cast<char*>(&numRows), sizeof(uint32_t));  // read number of rows
            std::getline(in_file, line, '\4');                                  // read control-D
            in_file.read(reinterpret_cast<char*>(&numCols), sizeof(uint32_t));  // read number of columns
            in_file.seekg(numRows * numCols * sizeof(float), in_file.cur);      // read data
            i++;
        }
        if (!in_file.eof()) {
            std::string line;
            std::getline(in_file, ptrName, '\0');  // read variable length name followed by space and NUL
            std::getline(in_file, line, '\4');     // read "BFM" followed by space and control-D
            if (line.compare("BFM ") != 0) {
                throw std::runtime_error(std::string("Cannot find array specifier in file %s in load_file()!\n") +
                                         fileName);
            }
            in_file.read(reinterpret_cast<char*>(ptrNumRows), sizeof(uint32_t));     // read number of rows
            std::getline(in_file, line, '\4');                                       // read control-D
            in_file.read(reinterpret_cast<char*>(ptrNumColumns), sizeof(uint32_t));  // read number of columns
            in_file.read(reinterpret_cast<char*>(&memory.front()),
                         *ptrNumRows * *ptrNumColumns * sizeof(float));  // read array data
        }
        in_file.close();
    } else {
        throw std::runtime_error(std::string("Failed to open %s for reading in load_file()!\n") + fileName);
    }

    *ptrNumBytesPerElement = sizeof(float);
}

void ArkFile::save_file(const char* fileName,
                        bool shouldAppend,
                        std::string name,
                        const void* ptrMemory,
                        uint32_t numRows,
                        uint32_t numColumns) {
    std::ios_base::openmode mode = std::ios::binary;
    if (shouldAppend) {
        mode |= (std::ios::ate | std::ios::in);
    }
    std::ofstream out_file(fileName, mode);
    if (out_file.good()) {
        const auto curPtr = out_file.tellp();
        out_file.seekp(0);
        out_file.write(name.c_str(), name.length());  // write name
        out_file.write("\0", 1);
        out_file.write("BFM ", 4);
        out_file.write("\4", 1);
        out_file.write(reinterpret_cast<char*>(&numRows), sizeof(uint32_t));
        out_file.write("\4", 1);
        out_file.write(reinterpret_cast<char*>(&numColumns), sizeof(uint32_t));
        if (shouldAppend) {
            out_file.seekp(curPtr);
        }
        out_file.write(reinterpret_cast<const char*>(ptrMemory), 1 * numColumns * sizeof(float));
        out_file.close();
    } else {
        throw std::runtime_error(std::string("Failed to open %s for writing in save_file()!\n") + fileName);
    }
}
