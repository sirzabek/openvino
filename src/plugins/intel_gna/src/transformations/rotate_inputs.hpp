// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Transpose convolution inputs (HW->WH)
 * when it is directly connected to network inputs
 *
 * Searches for next pattern
 *     Any input layer
 *           |
 *   Reshape/FQ/Squeeze/Usqueeze
 *           |
 *        Convolution
 *
 *    And transforms to
 *     Any input layer
 *           |
 *       Transpose
 *           |
 *    Reshape/FQ/Squeeze/Usqueeze
 *           |
 *        Convolution
 */
class InsertConvolutionTransposeHW : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertConvolutionTransposeHW", "0");
    InsertConvolutionTransposeHW();
};

/**
 * @brief Transpose convolution inputs (HW->WH)
 * when it is directly connected to network inputs
 *
 * Searches for next pattern
 *     Any input layer
 *           |
 *   Reshape/FQ/Squeeze/Usqueeze
 *           |
 *     GroupConvolution
 *
 *    And transforms to
 *     Any input layer
 *           |
 *       Transpose
 *           |
 *    Reshape/FQ/Squeeze/Usqueeze
 *           |
 *     GroupConvolution
 */
class InsertGroupConvolutionTransposeHW : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertGroupConvolutionTransposeHW", "0");
    InsertGroupConvolutionTransposeHW();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
