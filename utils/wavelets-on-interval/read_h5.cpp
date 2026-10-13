/**
 * @file read_h5.cpp
 * @brief Simple example: open HDF5 file and read 1D vector datasets
 *
 * Usage: ./read_h5 <filename.h5> [dataset_name] [slice_index]
 *   - filename.h5: path to HDF5 file (required)
 *   - dataset_name: name of dataset to read, e.g., "u" (default: "u")
 *   - slice_index: row index for 2D datasets (default: 0)
 */

#include <hdf5.h>

#include <algorithm>
#include <cmath>  //trunc
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "fwt01.h"

/**
 * @brief Load a 1D vector from an HDF5 dataset
 *
 * Supports both 1D datasets and 2D datasets (reads a single row slice).
 *
 * @param filename Path to the HDF5 file
 * @param dataset_name Name of the dataset (e.g., "u", "v", "w")
 * @param slice_index Row index to extract (only used for 2D datasets)
 * @param out_data Output vector to store the loaded data
 * @return true if loading succeeded, false otherwise
 */
bool load_1d_vector(const std::string& filename, const std::string& dataset_name, int slice_index, std::vector<float>& out_data)
{
    // Open the HDF5 file
    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error: Cannot open file '" << filename << "'" << std::endl;
        return false;
    }

    // Open the dataset
    hid_t dataset_id = H5Dopen2(file_id, dataset_name.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error: Dataset '" << dataset_name << "' not found in file" << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // Get dataspace and dimensionality
    hid_t file_space_id = H5Dget_space(dataset_id);
    int ndims = H5Sget_simple_extent_ndims(file_space_id);

    hsize_t dims[2] = {0, 0};
    H5Sget_simple_extent_dims(file_space_id, dims, nullptr);

    herr_t status = -1;

    if (ndims == 1) {
        // 1D dataset: read entire array
        out_data.resize(dims[0]);
        hid_t mem_space_id = H5Screate_simple(1, dims, nullptr);
        status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, mem_space_id, file_space_id, H5P_DEFAULT, out_data.data());
        H5Sclose(mem_space_id);
        std::cout << "Loaded 1D dataset '" << dataset_name << "' with " << dims[0] << " elements" << std::endl;
    } else if (ndims == 2) {
        // 2D dataset: extract a single row (slice)
        if (slice_index < 0 || slice_index >= static_cast<int>(dims[0])) {
            std::cerr << "Error: slice_index " << slice_index << " out of bounds [0, " << dims[0] << "]" << std::endl;
            H5Sclose(file_space_id);
            H5Dclose(dataset_id);
            H5Fclose(file_id);
            return false;
        }

        hsize_t row_length = dims[1];
        out_data.resize(row_length);

        // Select hyperslab: one row at slice_index
        hsize_t offset[2] = {static_cast<hsize_t>(slice_index), 0};
        hsize_t count[2] = {1, row_length};
        H5Sselect_hyperslab(file_space_id, H5S_SELECT_SET, offset, nullptr, count, nullptr);

        // Create 1D memory space
        hid_t mem_space_id = H5Screate_simple(1, &row_length, nullptr);
        status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, mem_space_id, file_space_id, H5P_DEFAULT, out_data.data());
        H5Sclose(mem_space_id);

        std::cout << "Loaded row " << slice_index << " from 2D dataset '" << dataset_name << "' with " << row_length << " elements" << std::endl;
    } else {
        std::cerr << "Error: Unsupported dataset dimensionality: " << ndims << "D" << std::endl;
        H5Sclose(file_space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    // Cleanup
    H5Sclose(file_space_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    return status >= 0;
}

/**
 * @brief List all datasets in an HDF5 file
 */
int list_datasets_callback(hid_t loc_id, const char* name, const H5L_info_t* info, void* op_data)
{
    auto* datasets = reinterpret_cast<std::vector<std::string>*>(op_data);

    H5O_info_t obj_info;
    herr_t status = H5Oget_info_by_name(loc_id, name, &obj_info, H5P_DEFAULT);
    if (status >= 0 && obj_info.type == H5O_TYPE_DATASET) {
        datasets->push_back(name);
    }
    return 0;
}

std::vector<std::string> list_datasets(const std::string& filename)
{
    std::vector<std::string> datasets;

    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        return datasets;
    }

    H5Literate(file_id, H5_INDEX_NAME, H5_ITER_INC, nullptr, list_datasets_callback, &datasets);

    H5Fclose(file_id);
    return datasets;
}

void print_usage(const char* program_name)
{
    std::cout << "Usage: " << program_name << " <filename.h5> [dataset_name] [slice_index]\n"
              << "\n"
              << "Arguments:\n"
              << "  filename.h5    Path to HDF5 file containing 1D vector datasets\n"
              << "  dataset_name   Name of dataset to read (default: \"u\")\n"
              << "  slice_index    Row index for 2D datasets (default: 0)\n"
              << "\n"
              << "Examples:\n"
              << "  " << program_name << " result.h5\n"
              << "  " << program_name << " result.h5 u\n"
              << "  " << program_name << " result.h5 v 5\n"
              << std::endl;
}

int main(int argc, char* argv[])
{
    // Parse command-line arguments
    if (argc < 2 || argc > 4) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::string filename = argv[1];
    std::string dataset_name = (argc >= 3) ? argv[2] : "u";
    int slice_index = (argc >= 4) ? std::atoi(argv[3]) : 0;

    std::cout << "HDF5 1D Vector Loader\n"
              << "=====================\n"
              << "File: " << filename << "\n"
              << "Dataset: " << dataset_name << "\n"
              << "Slice index: " << slice_index << "\n"
              << std::endl;

    // Load the data
    std::vector<float> data;
    if (!load_1d_vector(filename, dataset_name, slice_index, data)) {
        std::cerr << "\nFailed to load data. Available datasets in file:" << std::endl;
        std::vector<std::string> datasets = list_datasets(filename);
        if (datasets.empty()) {
            std::cerr << "  (none found or file not accessible)" << std::endl;
        } else {
            for (const auto& ds : datasets) {
                std::cerr << "  - " << ds << std::endl;
            }
        }
        return EXIT_FAILURE;
    }

    // Display sample data
    std::cout << "\nData preview (first 10 elements):\n";
    size_t preview_size = std::min(data.size(), static_cast<size_t>(10));
    std::cout << "  [";
    for (size_t i = 0; i < preview_size; ++i) {
        std::cout << data[i];
        if (i < preview_size - 1) std::cout << ", ";
    }
    if (data.size() > 10) std::cout << ", ...";
    std::cout << "]" << std::endl;

    // for (size_t i = 0; i < data.size(); ++i) std::cout << data[i]<<std::endl;
    std::cout << "\nStatistics:\n"
              << "  Size: " << data.size() << " elements\n"
              << "  Min: " << *std::min_element(data.begin(), data.end()) << "\n"
              << "  Max: " << *std::max_element(data.begin(), data.end()) << std::endl;
    //===== do wavelet transform ======//
    int MM = 4;
    int N_steps = round(trunc(std::log2(data.size() / (2 * MM))));
    std::vector<double> EE(N_steps);
    EE = fwt01(true, N_steps, MM, &data);
    std::cout << "fwt done" << std::endl;
    for (int i = 0; i < N_steps; ++i) std::cout << i + 1 << " " << EE[i] << std::endl;
    //=====================================//
    return EXIT_SUCCESS;
}
