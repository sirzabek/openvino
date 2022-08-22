// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "subgraph_tests/transpose_add.hpp"
#include "common_test_utils/test_constants.hpp"
namespace {
std::vector<std::pair<std::vector<size_t>, ngraph::Shape>> input_shapes {
    // Type 1 cases (see decompose_transpose.cpp in GNA Plugin)

    // GNA-compatible transpose
    {{1, 2, 4, 32}, {0, 3, 1, 2}},
    {{1, 2, 8, 8}, {0, 3, 1, 2}},
    {{2, 4, 32}, {2, 0, 1}},
    {{2, 16, 8}, {2, 0, 1}},
    {{1, 2, 4, 4}, {0, 2, 3, 1}},
    {{1, 16, 4, 2}, {0, 2, 3, 1}},
    {{1, 2, 2, 32}, {0, 2, 3, 1}},
    {{2, 4, 4}, {1, 2, 0}},
    {{16, 4, 2}, {1, 2, 0}},
    {{2, 2, 32}, {1, 2, 0}},


    // GNA-incompatible transpose
    {{1, 1, 32, 64}, {0, 1, 3, 2}},
    {{1, 64, 32}, {0, 2, 1}},
    {{128, 32}, {1, 0}},


    // Type 2 cases (see decompose_transpose.cpp in GNA Plugin)
    {{1, 16, 8, 32}, {0, 2, 1, 3}},
    {{4, 8, 64}, {1, 0, 2}},


    // Type 3 cases (see decompose_transpose.cpp in GNA Plugin)
    {{1, 2, 4, 32}, {0, 1, 3, 2}},
    {{4, 8, 32}, {0, 2, 1}},


    // GNA natively supported transposes
    {{1, 1, 4, 32}, {0, 1, 3, 2}},
    {{1, 4, 32}, {0, 2, 1}},
    {{4, 32}, {1, 0}},
    {{1, 4, 32}, {2, 0, 1}},
    {{1, 4, 32}, {1, 2, 0}},
    {{1, 4, 32}, {}},
    {{1, 8, 8},{}},
    {{1, 7, 8},{}},
    {{1, 40, 3},{}}
};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

std::map<std::string, std::string> additional_config = {
    {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
};
} // namespace

namespace SubgraphTestsDefinitions {
    INSTANTIATE_TEST_SUITE_P(smoke_basic, TransposeAdd,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GNA),
            ::testing::ValuesIn(input_shapes),
            ::testing::Values(additional_config)),
        TransposeAdd::getTestCaseName);
} // namespace SubgraphTestsDefinitions
