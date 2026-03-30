#pragma once

#include <hdf5.h>

#include <string>
#include <vector>

#include "params.h"

// ============================================================================
// CONFIGURATION: Toggle which components to save
// ============================================================================
// Set to 1 to save the component, 0 to skip it
#define SAVE_U_COMPONENT 1
#define SAVE_V_COMPONENT 0  // Set to 1 to save v component
#define SAVE_W_COMPONENT 0  // Set to 1 to save w component
// ============================================================================

/**
 * @brief HDF5 file writer for simulation output.
 *
 * Writes time-series data for the three-component system (u, v, w) to an HDF5 file.
 * Supports both 1D and 2D simulations with chunked storage and compression.
 *
 * File structure:
 * - /u: dataset of shape (num_snapshots, N) for 1D or (num_snapshots, N, N) for 2D (if SAVE_U_COMPONENT=1)
 * - /v: dataset of shape (num_snapshots, N) for 1D or (num_snapshots, N, N) for 2D (if SAVE_V_COMPONENT=1)
 * - /w: dataset of shape (num_snapshots, N) for 1D or (num_snapshots, N, N) for 2D (if SAVE_W_COMPONENT=1)
 * - /: attributes with simulation metadata (parameters, timestamps, etc.)
 */
class HDF5Writer
{
   public:
    /**
     * @brief Constructs and opens an HDF5 file for writing.
     *
     * Creates datasets for u, v, w variables with chunked storage and deflate compression (level 6).
     * Writes simulation metadata as file attributes.
     *
     * @param filename Path to the output HDF5 file
     * @param N Grid size per dimension
     * @param total_snapshots Total number of snapshots to store
     * @param dim Spatial dimension (1 or 2)
     * @throws std::runtime_error if file creation fails
     */
    HDF5Writer(const std::string& filename, int N, int total_snapshots, int dim);

    /**
     * @brief Destructor that closes all HDF5 handles.
     */
    ~HDF5Writer();

    /**
     * @brief Writes a single time step snapshot to the HDF5 file.
     *
     * Writes u, v, w data at the specified snapshot index using hyperslab selection.
     *
     * @param step_idx Snapshot index (0-based)
     * @param h_u Host array for activator variable u
     * @param h_v Host array for inhibitor variable v
     * @param h_w Host array for inhibitor variable w
     */
    void writeStep(int step_idx, const float* h_u, const float* h_v, const float* h_w);

    /**
     * @brief Write simulation metadata as HDF5 attributes
     *
     * Stores all simulation parameters for reproducibility.
     *
     * @param params Simulation parameters to store
     */
    void writeMetadata(const SimParams& params);

   private:
    hid_t file_id_;    ///< HDF5 file identifier
#if SAVE_U_COMPONENT
    hid_t dataset_u_;  ///< Dataset identifier for variable u
#endif
#if SAVE_V_COMPONENT
    hid_t dataset_v_;  ///< Dataset identifier for variable v
#endif
#if SAVE_W_COMPONENT
    hid_t dataset_w_;  ///< Dataset identifier for variable w
#endif
    hid_t filespace_;  ///< File dataspace identifier
    hid_t memspace_;   ///< Memory dataspace identifier

    int N_;                ///< Grid size per dimension
    int total_snapshots_;  ///< Total number of snapshots
    int dim_;              ///< Spatial dimension (1 or 2)

    /**
     * @brief Write a string attribute to the file
     */
    void writeAttribute(const std::string& name, const std::string& value);

    /**
     * @brief Write a float attribute to the file
     */
    void writeAttribute(const std::string& name, float value);

    /**
     * @brief Write an int attribute to the file
     */
    void writeAttribute(const std::string& name, int value);
};
