// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

class GnaFuseMarkUpNodesOrder : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("GnaFuseMarkUpNodesOrder", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

class GnaFuseCleanUpNodesOrder : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("GnaFuseCleanUpNodesOrder", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

class FuseGnaConvWithBiasAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseGnaConvWithBiasAdd", "0");
    FuseGnaConvWithBiasAdd();
};

class FuseGnaConvWithBiasAddAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseGnaConvWithBiasAddAdd", "0");
    FuseGnaConvWithBiasAddAdd();
};

class FuseGnaConvWithActivation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseGnaConvWithActivation", "0");
    FuseGnaConvWithActivation();
};

class FuseGnaDwscWithBiasAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseGnaDwscWithBiasAdd", "0");
    FuseGnaDwscWithBiasAdd();
};

class FuseGnaDwscWithBiasAddAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseGnaDwscWithBiasAddAdd", "0");
    FuseGnaDwscWithBiasAddAdd();
};

class FuseGnaDwscWithActivation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseGnaDwscWithActivation", "0");
    FuseGnaDwscWithActivation();
};

class GnaConvolutionFusion : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("GnaConvolutionFusion", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
