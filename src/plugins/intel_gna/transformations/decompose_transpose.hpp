// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief Decompose Transpose operation
 * Some types of Transpose operations which are not supported
 * natively by GNA HW are handled by this decomposition.
 * It employs several different methods, depending on the transposition order.
 * 
 */
class DecomposeTranspose : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("DecomposeTranspose", "0");
    DecomposeTranspose();
};

}  // namespace GNAPluginNS
