// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <time.h>

#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// clang-format off
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gna/properties.hpp>

#include <samples/args_helper.hpp>
#include <samples/slog.hpp>

#include "fileutils.hpp"
#include "speech_sample.hpp"
#include "utils.hpp"
// clang-format on

using namespace ov::preprocess;

/**
 * @brief The entry point for OpenVINO Runtime automatic speech recognition sample
 * @file speech_sample/main.cpp
 * @example speech_sample/main.cpp
 */
int main(int argc, char* argv[]) {
    try {
        // ------------------------------ Get OpenVINO Runtime version ----------------------------------------------
        slog::info << "OpenVINO runtime: " << ov::get_openvino_version() << slog::endl;

        // ------------------------------ Parsing and validation of input arguments ---------------------------------
        if (!parse_and_check_command_line(argc, argv)) {
            return 0;
        }

        // --------------------------- Step 1. Initialize OpenVINO Runtime core and read model
        // -------------------------------------
        ov::Core core;
        slog::info << "Loading model files:" << slog::endl << FLAGS_m << slog::endl;
        uint32_t batch_size = (FLAGS_cw_r > 0 || FLAGS_cw_l > 0) ? 1 : (uint32_t)FLAGS_bs;
        std::shared_ptr<ov::Model> model;
        std::vector<std::string> outputs;
        std::vector<size_t> ports;
        // --------------------------- Processing custom outputs ---------------------------------------------
        if (!FLAGS_oname.empty()) {
            std::vector<std::string> output_names = convert_str_to_vector(FLAGS_oname);
            for (const auto& output_name : output_names) {
                auto pos_layer = output_name.rfind(":");
                if (pos_layer == std::string::npos) {
                    throw std::logic_error("Output " + output_name + " doesn't have a port");
                }
                outputs.push_back(output_name.substr(0, pos_layer));
                try {
                    ports.push_back(std::stoi(output_name.substr(pos_layer + 1)));
                } catch (const std::exception&) {
                    throw std::logic_error("Ports should have integer type");
                }
            }
        }
        }
        // ------------------------------ Preprocessing ------------------------------------------------------
        // the preprocessing steps can be done only for loaded network and are not applicable for the imported network
        // (already compiled)
        if (!FLAGS_m.empty()) {
            model = core.read_model(FLAGS_m);
            if (!outputs.empty()) {
                // add custom output
                for (size_t i = 0; i < outputs.size(); i++) {
                    auto output = model->add_output(outputs[i], ports[i]);
                    output.add_names({outputs[i], outputs[i] + ":" + std::to_string(ports[i])});
                }
                // clean up other outputs
                ov::ResultVector results_to_remove;
                for (auto result : model->get_results()) {
                    bool is_name_found = false;
                    for (const std::string &name : result->get_input_tensor(0).get_names()) {
                        if (std::count(outputs.begin(), outputs.end(), name) != 0) {
                            is_name_found = true;
                            break;
                        }
                    }
                    if (!is_name_found) {
                        results_to_remove.push_back(result);
                    }
                }
                for (auto result : results_to_remove) {
                    model->remove_result(result);
                }
            }
            ov::preprocess::PrePostProcessor proc(model);
            const auto& inputs = model->inputs();
                proc.input(i).tensor().set_element_type(ov::element::f32);
            if (!FLAGS_layout.empty()) {
                custom_layouts = parse_input_layouts(FLAGS_layout, inputs);
            }
            for (const auto& input : inputs) {
                const auto& item_name = input.get_any_name();
                auto& in = proc.input(item_name);
                in.tensor().set_element_type(ov::element::f32);
                // Explicitly set inputs layout
                if (custom_layouts.count(item_name) > 0) {
                    in.model().set_layout(ov::Layout(custom_layouts.at(item_name)));
                }
            }
            for (int i = 0; i < model->outputs().size(); i++) {
                proc.output(i).tensor().set_element_type(ov::element::f32);
            }
            model = proc.build();
            // ov::set_batch(model, batch_size);
                if (FLAGS_layout.empty() &&
                    std::any_of(inputs.begin(), inputs.end(), [](const ov::Output<ov::Node>& i) {
                        return ov::layout::get_layout(i).empty();
                    })) {
                    throw std::logic_error(
                        "-bs option is set to " + std::to_string(FLAGS_bs) +
                        " but model does not contain layout information for any input. Please "
                        "specify it explicitly using -layout option. For example, input1[NCHW], input2[NC] or [NC]");
                } else {
                }
            }
        }
        // ------------------------------ Get Available Devices ------------------------------------------------------
        auto is_feature = [&](const std::string xFeature) {
            return FLAGS_d.find(xFeature) != std::string::npos;
        };
        bool use_gna = is_feature("GNA");
        bool use_hetero = is_feature("HETERO");
        std::string device_str = use_hetero && use_gna ? "HETERO:GNA,CPU" : FLAGS_d.substr(0, (FLAGS_d.find("_")));
        // -----------------------------------------------------------------------------------------------------
        // --------------------------- Set parameters and scale factors -------------------------------------
        struct UtteranceInfo {
            uint32_t num_frames = 0;            // num_rows
            uint32_t num_frame_elements = 0;    // num_cols
            uint32_t num_bytes_per_element = 0; // elem_size
            uint32_t num_bytes = 0;             // num_rows * num_cols * elem_size
            std::vector<uint8_t> data = {};
            uint8_t* data_ptr = nullptr;
        };

        struct FileInfo {
            std::string path = "";
            BaseFile* file_ptr = nullptr;
            uint32_t num_utterances = 0;
            uint32_t num_memory_bytes = 0;
            std::vector<UtteranceInfo> u_info = {};
        };

        ArkFile arkFile;
        NumpyFile numpyFile;
        auto get_file_ptr = [&](const std::string& filename) -> BaseFile* {
            auto ext = fileExt(FLAGS_i);
            if (ext == "ark") {
                return &arkFile;
            } else if (ext == "npz") {
                return &numpyFile;
            } else {
                throw std::logic_error("Invalid input file: " + filename);
            }
        };

        uint32_t (num_utterances)(0);
        auto fill_file_info = [&](const std::map<std::string, std::string> file_names_map,
                                  std::map<std::string, FileInfo>& file_info_map) {
            for (auto&& fn : file_names_map) {
                FileInfo file_info;
                file_info.path = fn.second;
                file_info.file_ptr = get_file_ptr(file_info.path);
                file_info_map[fn.first] = file_info;
                file_info_map[fn.first].u_info.resize(num_utterances);
            }
        };

        std::map<std::string, FileInfo> i_file_info_map;
        if (!FLAGS_i.empty()) {
            fill_file_info(parse_file_names(model->inputs(), FLAGS_i), i_file_info_map);
            for (auto&& i_file_info : i_file_info_map) {
                auto filename = i_file_info.second.path.c_str();
                auto file = i_file_info.second.file_ptr;
                file->get_file_info(filename, 0, &i_file_info.second.num_utterances, &i_file_info.second.num_memory_bytes);
                i_file_info.second.u_info.resize(i_file_info.second.num_utterances);
                if (num_utterances == 0) {
                    num_utterances = i_file_info.second.num_utterances;
                } else if (i_file_info.second.num_utterances != num_utterances) {
                    throw std::logic_error(
                        "Incorrect input files. Number of utterance must be the same for all input files");
                }
            }
        }
        /** Setting parameter for per layer metrics **/
        ov::AnyMap gna_plugin_config;
        ov::AnyMap generic_plugin_config;
        if (use_gna) {
            std::string gna_device =
                use_hetero ? FLAGS_d.substr(FLAGS_d.find("GNA"), FLAGS_d.find(",") - FLAGS_d.find("GNA")) : FLAGS_d;
            auto parse_gna_device = [&](const std::string& device) -> ov::intel_gna::ExecutionMode {
                ov::intel_gna::ExecutionMode mode;
                std::stringstream ss(device);
                ss >> mode;
                return mode;
            };
            gna_plugin_config[ov::intel_gna::execution_mode.name()] =
                gna_device.find("_") == std::string::npos ? ov::intel_gna::ExecutionMode::AUTO : parse_gna_device(gna_device);
                                                                        : parse_gna_device(gnaDevice);
        }
        if (FLAGS_pc) {
            generic_plugin_config[ov::enable_profiling.name()] = true;
        }
        if (FLAGS_q.compare("user") == 0) {
            if (!FLAGS_rg.empty()) {
                std::string errMessage("Custom scale factor can not be set for imported gna model: " + FLAGS_rg);
                throw std::logic_error(errMessage);
            auto scale_factors_per_input = parse_scale_factors(model->inputs(), FLAGS_sf);
            for (auto&& sf : scale_factors_per_input) {
                slog::info << "For input " << sf.first << " using scale factor of " << sf.second << slog::endl;
                                           " scale factors provided for " + std::to_string(numInputFiles) +
                                           " input files.");
            }
            gna_plugin_config[ov::intel_gna::scale_factors_per_input.name()] = scale_factors_per_input;
        } else {
            // "static" quantization with calculated scale factor
            if (!FLAGS_rg.empty()) {
                slog::info << "Using scale factor from provided imported gna model: " << FLAGS_rg << slog::endl;
            } else {
                std::map<std::string, float> scale_factors_per_input = {};
                for (auto&& i_file_info : i_file_info_map) {
                    std::string name;
                    std::vector<uint8_t> ptr_features;
                    uint32_t num_arrays(0), num_bytes(0), num_frames(0), num_frame_elements(0), num_bytes_ber_element(0);
                    auto file = i_file_info.second.file_ptr;
                    auto filename = i_file_info.second.path.c_str();
                    file->get_file_info(filename, 0, &num_arrays, &num_bytes);
                    ptr_features.resize(num_bytes);
                    file->load_file(filename,
                                    0,
                                    name,
                                    ptr_features,
                                    &num_frames,
                                    &num_frame_elements,
                                    &num_bytes_ber_element);
                    auto scale_factor = scale_factor_for_quantization(ptr_features.data(),
                                                                      MAX_VAL_2B_FEAT,
                                                                      num_frames * num_frame_elements);
                    slog::info << "Using scale factor of " << scale_factor << " calculated from first utterance."
                               << slog::endl;
                    scale_factors_per_input[i_file_info.first] = scale_factor;
                }
                gna_plugin_config[ov::intel_gna::scale_factors_per_input.name()] = scale_factors_per_input;
            }
        }
        gna_plugin_config[ov::hint::inference_precision.name()] = (FLAGS_qb == 8) ? ov::element::i8 : ov::element::i16;
        auto parse_target = [&](const std::string& target) -> ov::intel_gna::HWGeneration {
            return (target == "GNA_TARGET_2_0") ? ov::intel_gna::HWGeneration::GNA_2_0 :
                   (target == "GNA_TARGET_3_0") ? ov::intel_gna::HWGeneration::GNA_3_0 :
                   ov::intel_gna::HWGeneration::UNDEFINED;
        };
        gna_plugin_config[ov::intel_gna::execution_target.name()] = parse_target(FLAGS_exec_target);
        gna_plugin_config[ov::intel_gna::compile_target.name()] = parse_target(FLAGS_compile_target);
        gna_plugin_config[ov::intel_gna::memory_reuse.name()] = false;
        gna_plugin_config[ov::intel_gna::pwl_max_error_percent.name()] = FLAGS_pwl_me;
        // -----------------------------------------------------------------------------------------------------
        // --------------------------- Write model to file --------------------------------------------------
        // Embedded GNA model dumping (for Intel(R) Speech Enabling Developer Kit)
        if (!FLAGS_we.empty()) {
            gna_plugin_config[ov::intel_gna::firmware_model_image_path.name()] = FLAGS_we;
        }
        // -----------------------------------------------------------------------------------------------------
        // --------------------------- Step 2. Loading model to the device ------------------------------------------
        if (use_gna) {
            generic_plugin_config.insert(std::begin(gna_plugin_config), std::end(gna_plugin_config));
        }
        auto t0 = Time::now();
        ms loadTime = std::chrono::duration_cast<ms>(Time::now() - t0);
        slog::info << "Model loading time " << loadTime.count() << " ms" << slog::endl;
        ov::CompiledModel compiled_model;
        if (!FLAGS_m.empty()) {
            slog::info << "Loading model to the device " << FLAGS_d << slog::endl;
            compiled_model = core.compile_model(model, device_str, generic_plugin_config);
        } else {
            slog::info << "Importing model to the device" << slog::endl;
            std::ifstream streamrq(FLAGS_rg, std::ios_base::binary | std::ios_base::in);
            if (!streamrq.is_open()) {
                throw std::runtime_error("Cannot open model file " + FLAGS_rg);
            }
            compiled_model = core.import_model(streamrq, device_str, generic_plugin_config);
            // loading batch from exported model
            const auto& imported_inputs = executableNet.inputs();
            if (std::any_of(imported_inputs.begin(), imported_inputs.end(), [](const ov::Output<const ov::Node>& i) {
                    return ov::layout::get_layout(i).empty();
                })) {
                slog::warn << "No batch dimension was found at any input, assuming batch to be 1." << slog::endl;
                batchSize = 1;
            } else {
                for (auto& info : imported_inputs) {
                    auto imported_layout = ov::layout::get_layout(info);
                    if (ov::layout::has_batch(imported_layout)) {
                        batchSize = (uint32_t)info.get_shape()[ov::layout::batch_idx(imported_layout)];
                        break;
                    }
                }
            }
        }
        // --------------------------- Exporting gna model using InferenceEngine AOT API---------------------
        if (!FLAGS_wg.empty()) {
            slog::info << "Writing GNA Model to file " << FLAGS_wg << slog::endl;
            t0 = Time::now();
            std::ofstream streamwq(FLAGS_wg, std::ios_base::binary | std::ios::out);
            compiled_model.export_model(streamwq);
            ms exportTime = std::chrono::duration_cast<ms>(Time::now() - t0);
            slog::info << "Exporting time " << exportTime.count() << " ms" << slog::endl;
            return 0;
        }
        if (!FLAGS_we.empty()) {
            slog::info << "Exported GNA embedded model to file " << FLAGS_we << slog::endl;
            return 0;
        }
        // ---------------------------------------------------------------------------------------------------------
        // --------------------------- Step 3. Create infer request
        // --------------------------------------------------
        std::vector<InferRequestStruct> inferRequests(1);

        for (auto& inferRequest : inferRequests) {
            inferRequest = {compiled_model.create_infer_request(), -1, batch_size};
        }
        // --------------------------- Step 4. Configure input & output
        // --------------------------------------------------
        std::map<std::string, ov::Tensor> input_tensors_map;
        auto inputs = compiled_model.inputs();
        check_number_of_files(inputs.size(), i_file_info_map.size());
        if (!FLAGS_iname.empty()) {
            std::vector<std::string> inputNameBlobs = convert_str_to_vector(FLAGS_iname);
            for (const auto& input_name : inputNameBlobs) {
                // check_name(inputs, input_name);
                ov::Tensor tensor = inferRequests.begin()->inferRequest.get_tensor(input_name);
                if (!tensor) {
                    std::string errMessage("No tensor with name : " + input_name);
                    throw std::logic_error(errMessage);
                }
                input_tensors_map[input_name] = tensor;
            }
        } else {
            for (const auto& input : inputs) {
                input_tensors_map[input.get_any_name()] = inferRequests.begin()->inferRequest.get_tensor(input);
            }
        }

        std::map<std::string, FileInfo> o_file_info_map;
        std::map<std::string, FileInfo> r_file_info_map;

        if (!FLAGS_o.empty()) {
            fill_file_info(parse_file_names(model->outputs(), FLAGS_o), o_file_info_map);
        }
        if (!FLAGS_r.empty()) {
            fill_file_info(parse_file_names(model->outputs(), FLAGS_r), r_file_info_map);
        }

        auto get_num_scores_per_frame = [&](const std::string& name) -> const size_t {
            auto dims = compiled_model.output(name).get_shape();
            return std::accumulate(std::begin(dims), std::end(dims), size_t{1}, std::multiplies<size_t>());
        };
        // -----------------------------------------------------------------------------------------------------
        // --------------------------- Step 5. Do inference --------------------------------------------------------
        ScoreErrorT frame_error, total_error;
                                                                       : outputs.size());
        std::vector<std::vector<uint8_t>> vectorPtrReferenceScores(reference_name_files.size());
        std::vector<ScoreErrorT> vectorFrameError(reference_name_files.size()),
            vectorTotalError(reference_name_files.size());
        // initialize memory state before starting
        for (auto&& state : inferRequests.begin()->inferRequest.query_state()) {
            state.reset();
        }
        /** Work with each utterance **/
        for (uint32_t utt_i = 0; utt_i < num_utterances; ++utt_i) {
            std::map<std::string, ov::ProfilingInfo> utterance_perf_map;
            uint64_t total_number_of_runs_on_hw = 0;
            std::string utt_name;
            uint32_t num_frames(0);

            auto get_utt_data = [ &utt_i ] (FileInfo& file_info, std::string& utt_name) -> void {
                uint32_t n(0);
                auto filename = file_info.path.c_str();
                auto file_ptr = file_info.file_ptr;
                file_ptr->get_file_info(filename, utt_i, &n, &file_info.u_info[utt_i].num_bytes);
                file_info.u_info[utt_i].data.resize(file_info.u_info[utt_i].num_bytes);
                file_ptr->load_file(filename,
                                    utt_i,
                                    utt_name,
                                    file_info.u_info[utt_i].data,
                                    &file_info.u_info[utt_i].num_frames,
                                    &file_info.u_info[utt_i].num_frame_elements,
                                    &file_info.u_info[utt_i].num_bytes_per_element);
            };

            /** Get information from input file for current utterance **/
            for (auto&& i_file_info : i_file_info_map) {
                get_utt_data(i_file_info.second, utt_name);
                if (num_frames == 0) {
                    num_frames = i_file_info.second.u_info[utt_i].num_frames;
                } else if (num_frames != i_file_info.second.u_info[utt_i].num_frames) {
                    std::string errMessage("Number of frames in input files is different: " +
                                            std::to_string(num_frames) + " and " + std::to_string(i_file_info.second.u_info[utt_i].num_frames));
                    throw std::logic_error(errMessage);
                }
                auto tensor_size = input_tensors_map.at(i_file_info.first).get_size();
                auto num_elements = i_file_info.second.u_info[utt_i].num_frame_elements * batch_size;
                if (tensor_size != num_elements) {
                    throw std::logic_error("network input size(" + std::to_string(tensor_size) +
                                            ") mismatch to input file size (" + std::to_string(num_elements) + ")");
            }
                i_file_info.second.u_info[utt_i].data_ptr = &i_file_info.second.u_info[utt_i].data.front();
            }
            /** Resize output data array **/
            if (!FLAGS_o.empty()) {
                for (auto& o_file_info : o_file_info_map) {
                    auto num_scores_per_frame = get_num_scores_per_frame(o_file_info.first);
                    slog::info << "Number scores per frame : " << num_scores_per_frame << slog::endl;
                    o_file_info.second.u_info[utt_i].data.resize(num_frames * num_scores_per_frame * sizeof(float));
                }
            }
            double totalTime = 0.0;

            for (size_t errorIndex = 0; errorIndex < vectorFrameError.size(); errorIndex++) {
                clear_score_error(&vectorTotalError[errorIndex]);
                vectorTotalError[errorIndex].threshold = vectorFrameError[errorIndex].threshold = MAX_SCORE_DIFFERENCE;
            }

            std::vector<uint8_t*> inputFrame;
            for (auto& ut : ptrUtterances) {
                inputFrame.push_back(&ut.front());
            }
            std::map<std::string, ov::ProfilingInfo> callPerfMap;
            size_t frameIndex = 0;
            uint32_t numFramesFile = numFrames;
            numFrames += FLAGS_cw_l + FLAGS_cw_r;
            uint32_t numFramesThisBatch{batchSize};
            auto t0 = Time::now();
            auto t1 = t0;

            BaseFile* fileReferenceScores;
            std::string refUtteranceName;

            if (!reference_data.first.empty()) {
                /** Read file with reference scores **/
                for (auto& r_file_info : r_file_info_map) {
                    std::string ref_utterance_name;
                    get_utt_data(r_file_info.second, ref_utterance_name);
                    if (fileReferenceScores != nullptr) {
               }
            }
            double total_time = 0.0;
            std::cout << "Utterance " << utt_i << ": " << std::endl;
            clear_score_error(&total_error);
            total_error.threshold = frame_error.threshold = MAX_SCORE_DIFFERENCE;
            std::map<std::string, ov::ProfilingInfo> call_perf_map;
            size_t frame_i = 0;
            uint32_t num_frames_file = num_frames;
            num_frames += FLAGS_cw_l + FLAGS_cw_r;
            uint32_t num_frames_this_batch{batch_size};
            auto t0 = Time::now();
            auto t1 = t0;
            while (frame_i <= num_frames) {
                if (frame_i == num_frames) {
                    if (std::find_if(inferRequests.begin(), inferRequests.end(), [&](InferRequestStruct x) {
                            return (x.frameIndex != -1);
                        }) == inferRequests.end()) {
                        break;
                    }
                }
                bool inferRequestFetched = false;
                /** Start inference loop **/
                for (auto& inferRequest : inferRequests) {
                    num_frames_this_batch = (frame_i == num_frames) ? 1 :
                            (num_frames - frame_i < batch_size) ? (num_frames - frame_i) : batch_size;

                    /* waits until inference result becomes available */
                    if (inferRequest.frameIndex != -1) {
                        inferRequest.inferRequest.wait();
                        if (inferRequest.frameIndex >= 0) {
                            // Get outputs data
                            for (auto&& o_file_info : o_file_info_map) {
                                auto output_name = o_file_info.first;
                                auto num_scores_per_frame = get_num_scores_per_frame(output_name);
                                auto dims = executableNet.output(outputName).get_shape();
                                numScoresPerOutput[next_output] = std::accumulate(std::begin(dims),
                                                                                  std::end(dims),
                                                                                  size_t{1},
                                                                                  std::multiplies<size_t>());

                                vectorPtrScores[next_output].resize(numFramesFile * numScoresPerOutput[next_output] *
                                                                    sizeof(float));

                                /* Prepare output data for save to file in future */
                                uint8_t* data_ptr = &o_file_info.second.u_info[utt_i].data.front() +
                                                num_scores_per_frame * sizeof(float) * (inferRequest.frameIndex);
                                ov::Tensor output_tensor = inferRequest.inferRequest.get_tensor(output_name);
                                // locked memory holder should be alive all time while access to its buffer happens
                                std::memcpy(data_ptr, output_tensor.data<float>(), num_scores_per_frame * sizeof(float));
                            }
                            /** Compare output data with reference scores **/
                            for (auto&& r_file_info : r_file_info_map) {
                                auto output_name = r_file_info.first;
                                ov::Tensor output_tensor =
                                    inferRequest.inferRequest.get_tensor(output_name);
                                compare_scores(
                                    output_tensor.data<float>(),
                                    &r_file_info.second.u_info[utt_i].data[inferRequest.frameIndex *
                                                        r_file_info.second.u_info[utt_i].num_frame_elements *
                                                        r_file_info.second.u_info[utt_i].num_bytes_per_element],
                                    &frame_error,
                                            &vectorFrameError[next_output],
                                    inferRequest.numFramesThisBatch,
                                    r_file_info.second.u_info[utt_i].num_frame_elements);
                                update_score_error(&frame_error, &total_error);
                                                           &vectorTotalError[next_output]);
                                    } else {
                                        throw std::logic_error("Number of output and reference frames does not match.");
                                    }
                            }
                            if (FLAGS_pc) {
                                // retrieve new counters
                                get_performance_counters(inferRequest.inferRequest, call_perf_map);
                                // summarize retrieved counters with all previous
                                sum_performance_counters(call_perf_map, utterance_perf_map, total_number_of_runs_on_hw);
                            }
                        }
                        // -----------------------------------------------------------------------------------------------------
                    }
                    if (frame_i == num_frames) {
                        inferRequest.frameIndex = -1;
                        continue;
                    }
                    ptrInputBlobs.clear();
                    if (input_data.second.empty()) {
                        for (auto& input : cInputInfo) {
                            ptrInputBlobs.push_back(inferRequest.inferRequest.get_tensor(input));
                        }
                    } else {
                        std::vector<std::string> inputNameBlobs = input_data.second;
                        for (const auto& input : inputNameBlobs) {
                            ov::Tensor blob = inferRequests.begin()->inferRequest.get_tensor(input);
                            if (!blob) {
                                std::string errMessage("No blob with name : " + input);
                                throw std::logic_error(errMessage);
                            }
                            ptrInputBlobs.push_back(blob);
                        }
                    }

                    /** Iterate over all the input blobs **/
                    for (size_t i = 0; i < numInputFiles; ++i) {
                        ov::Tensor minput = ptrInputBlobs[i];
                        if (!minput) {
                            std::string errMessage("We expect ptrInputBlobs[" + std::to_string(i) +
                                                   "] to be inherited from Tensor, " +
                                                   "but in fact we were not able to cast input to Tensor");
                            throw std::logic_error(errMessage);
                        }
                        memcpy(minput.data(),
                               inputFrame[i],
                               numFramesThisBatch * numFrameElementsInput[i] * sizeof(float));
                        // Used to infer fewer frames than the batch size
                        if (batchSize != numFramesThisBatch) {
                            memset(minput.data<float>() + numFramesThisBatch * numFrameElementsInput[i],
                                   0,
                                   (batchSize - numFramesThisBatch) * numFrameElementsInput[i]);
                        }
                    }
                    // -----------------------------------------------------------------------------------------------------
                    int index = static_cast<int>(frame_i) - (FLAGS_cw_l + FLAGS_cw_r);
                    // size_t i = 0;
                    for (auto&& i_file_info : i_file_info_map) {
                        // inferRequest.inferRequest.set_input_tensor(
                        //     i++,
                        //     ov::Tensor(ov::element::f32, input_tensors_map.at(i_file_info.first).get_shape(), i_file_info.second.u_info[utt_i].data_ptr));

                        inferRequest.inferRequest.set_tensor(i_file_info.first,
                            ov::Tensor(ov::element::f32, input_tensors_map.at(i_file_info.first).get_shape(), i_file_info.second.u_info[utt_i].data_ptr));
                    }
                    /* Starting inference in asynchronous mode*/
                    inferRequest.inferRequest.start_async();
                    inferRequest.frameIndex = index < 0 ? -2 : index;
                    inferRequest.numFramesThisBatch = num_frames_this_batch;
                    frame_i += num_frames_this_batch;
                    for (auto&& i_file_info : i_file_info_map) {
                        auto num_frame_elements = i_file_info.second.u_info[utt_i].num_frame_elements;
                        if (FLAGS_cw_l > 0 || FLAGS_cw_r > 0) {
                            int idx = frame_i - FLAGS_cw_l;
                            if (idx > 0 && idx < static_cast<int>(num_frames_file)) {
                                i_file_info.second.u_info[utt_i].data_ptr += sizeof(float) * num_frame_elements * num_frames_this_batch;
                            } else if (idx >= static_cast<int>(num_frames_file)) {
                                i_file_info.second.u_info[utt_i].data_ptr = &i_file_info.second.u_info[utt_i].data.front() + (num_frames_file - 1) * sizeof(float) *
                                                                                num_frame_elements *
                                                                                num_frames_this_batch;
                            } else if (idx <= 0) {
                                i_file_info.second.u_info[utt_i].data_ptr = &i_file_info.second.u_info[utt_i].data.front();
                            }
                        } else {
                            i_file_info.second.u_info[utt_i].data_ptr += sizeof(float) * num_frame_elements * num_frames_this_batch;
                        }
                    }
                    inferRequestFetched |= true;
                }
                /** Inference was finished for current frame **/
                if (!inferRequestFetched) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
            }
            t1 = Time::now();
            fsec fs = t1 - t0;
            ms d = std::chrono::duration_cast<ms>(fs);
            total_time += d.count();
            // resetting state between utterances
            for (auto&& state : inferRequests.begin()->inferRequest.query_state()) {
                state.reset();
            }
            // -----------------------------------------------------------------------------------------------------

            // --------------------------- Step 6. Process output
            // -------------------------------------------------------
            for (auto&& o_file_info : o_file_info_map) {
                auto num_scores_per_frame = get_num_scores_per_frame(o_file_info.first);
                auto file = o_file_info.second.file_ptr;
            std::cout << "Total time in Infer (HW and SW):\t" << totalTime << " ms" << std::endl;
            std::cout << "Frames in utterance:\t\t\t" << numFrames << " frames" << std::endl;
            std::cout << "Average Infer time per frame:\t\t" << totalTime / static_cast<double>(numFrames) << " ms\n"
                      << std::endl;

            if (FLAGS_pc) {
                // print performance results
                print_performance_counters(utterancePerfMap,
                                           frameIndex,
                                           std::cout,
                                           getFullDeviceName(core, FLAGS_d),
                                           totalNumberOfRunsOnHw,
                                           FLAGS_d);
            }

            for (size_t next_output = 0; next_output < count_file; next_output++) {
                /* Save output data to file */
                bool shouldAppend = (utt_i == 0) ? false : true;
                file->save_file(o_file_info.second.path.c_str(),
                                shouldAppend,
                                utt_name,
                                &o_file_info.second.u_info[utt_i].data.front(),
                                num_frames_file,
                                num_scores_per_frame);
            }
            /** Show performance results **/
            std::cout << "Total time in Infer (HW and SW):\t" << total_time << " ms" << std::endl;
            std::cout << "Frames in utterance:\t\t\t" << num_frames << " frames" << std::endl;
            std::cout << "Average Infer time per frame:\t\t" << total_time / static_cast<double>(num_frames) << " ms"
                        << std::endl;
            if (FLAGS_pc) {
                // print performance results
                print_performance_counters(utterance_perf_map,
                                            frame_i,
                                            std::cout,
                                            getFullDeviceName(core, FLAGS_d),
                                            total_number_of_runs_on_hw,
                                            FLAGS_d);
            }
            if (!FLAGS_r.empty()) {
                // print statistical score error
                print_reference_compare_results(total_error, num_frames, std::cout);
                                                                   : output_names[next_output];
                    std::cout << "Output name: " << outputName << std::endl;
                    std::cout << "Number scores per frame: " << numScoresPerOutput[next_output] / batchSize << std::endl
                              << std::endl;
                    print_reference_compare_results(vectorTotalError[next_output], numFrames, std::cout);
            }
            std::cout << "End of Utterance " << utt_i << std::endl << std::endl;
            // -----------------------------------------------------------------------------------------------------
        }
    } catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    } catch (...) {
        slog::err << "Unknown/internal exception happened" << slog::endl;
        return 1;
    }
    slog::info << "Execution successful" << slog::endl;
    return 0;
}
