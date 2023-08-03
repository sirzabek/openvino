// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

class InsertTransposeBeforeMatMul : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertTransposeBeforeMatMul", "0");
    InsertTransposeBeforeMatMul();
};

class ReplaceTransposeBeforeMatMul : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReplaceTransposeBeforeMatMul", "0");
    ReplaceTransposeBeforeMatMul();
};

class RemoveTransposeBeforeAdd : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("RemoveTransposeBeforeAdd", "0");
    RemoveTransposeBeforeAdd();
};

class InsertTransposeBeforeMultiply : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertTransposeBeforeMultiply", "0");
    InsertTransposeBeforeMultiply();
};

class InsertPreprocessingTranspose : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertPreprocessingTranspose", "0");
    InsertPreprocessingTranspose();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
