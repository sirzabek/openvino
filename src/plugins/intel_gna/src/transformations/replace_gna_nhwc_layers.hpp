// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Substitites ngraph::Convolution (NCHW) -> GNAConvolution (NHWC)
 *
 *                              Transpose (NCHW -> NHWC)
 *                                       |
 * Convolution (NCHW) ->         GNAConvolution (NHWC)
 *                                       |
 *                              Transpose (NHWC -> NCHW)
 */
class SubstituteConvolution : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SubstituteConvolution();
};

/**
 * @brief Substitites ngraph::GroupConvolution (NCHW) -> GNADwsc (NHWC)
 *
 *                              Transpose (NCHW -> NHWC)
 *                                       |
 * GroupConvolution (NCHW) ->        GNADwsc (NHWC)
 *                                       |
 *                              Transpose (NHWC -> NCHW)
 */
class SubstituteGroupConvolution : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SubstituteGroupConvolution();
};

/**
 * @brief Substitites ngraph::MaxPool (NCHW) -> GNAPool (NHWC)
 *
 *                              Transpose (NCHW -> NHWC)
 *                                       |
 * MaxPool (NCHW) ->               GNAPool (NHWC)
 *                                       |
 *                              Transpose (NHWC -> NCHW)
 */
class SubstituteMaxPool : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SubstituteMaxPool();
};

/**
 * @brief Substitites ngraph::MaxPool (NCHW) -> GNAPool (NHWC)
 *
 *                              Transpose (NCHW -> NHWC)
 *                                       |
 * MaxPool (NCHW) ->               GNAPool (NHWC)
 *                                       |
 *                              Transpose (NHWC -> NCHW)
 */
class SubstituteAvgPool : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SubstituteAvgPool();
};

/**
 * @brief calls SubstituteConvolution, SubstituteGNADwsc and SubstituteGNAPool together
 */
class ReplaceGnaNHWCLayers : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_model(const std::shared_ptr<ngraph::Function>& f) override;
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
