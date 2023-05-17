// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/handle_matmul_reshape_transpose.hpp"

#include <algorithm>
#include <memory>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <vector>

#include "openvino/cc/pass/itt.hpp"

using ngraph::Node;
using ngraph::opset9::Add;
using ngraph::opset9::Concat;
using ngraph::opset9::Constant;
using ngraph::opset9::Convolution;
using ngraph::opset9::FakeQuantize;
using ngraph::opset9::MatMul;
using ngraph::opset9::Multiply;
using ngraph::opset9::Reshape;
using ngraph::opset9::Sigmoid;
using ngraph::opset9::StridedSlice;
using ngraph::opset9::Transpose;

namespace {

std::shared_ptr<Constant> make_constant(const std::vector<size_t>& data) {
    return std::make_shared<Constant>(ngraph::element::Type_t::i64, ngraph::Shape{data.size()}, data);
}

std::shared_ptr<Reshape> append_reshape(std::shared_ptr<Node> node, const std::vector<size_t>& data) {
    auto const_shape = make_constant(data);
    auto r = std::make_shared<Reshape>(node, const_shape, false);
    return r;
}

std::shared_ptr<Node> append_simple_transpose(std::shared_ptr<Node> node) {
    auto order = make_constant(ov::Shape{1, 0});
    auto t = std::make_shared<Transpose>(node, order);
    return t;
}

}  // namespace

namespace ov {
namespace intel_gna {
namespace pass {
using ngraph::pattern::any_input;
using ngraph::pattern::Matcher;
using ngraph::pattern::wrap_type;

InsertTransposeBeforeMatMul::InsertTransposeBeforeMatMul() {
    MATCHER_SCOPE(InsertTransposeBeforeMatMul);

    auto constant_pattern = wrap_type<Constant>();
    auto fq = wrap_type<FakeQuantize>({any_input(), any_input(), any_input(), any_input(), any_input()});
    auto reshape0 = wrap_type<Reshape>({fq, constant_pattern});
    auto reshape1 = wrap_type<Reshape>({reshape0, constant_pattern});
    auto reshape2 = wrap_type<Reshape>({reshape1, constant_pattern});
    auto matmul = wrap_type<MatMul>({reshape2, any_input()});

    matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        // Add transpose
        auto fq_node = pattern_map.at(fq).get_node_shared_ptr();
        auto reshape0_node = pattern_map.at(reshape0).get_node_shared_ptr();
        auto consumers = reshape0_node->output(0).get_target_inputs();
        auto shape = reshape0_node->get_output_shape(0);
        std::swap(shape.at(2), shape.at(3));
        auto reshape_node_new = append_reshape(fq_node, shape);
        auto order = make_constant(ov::Shape{0, 1, 3, 2});
        auto transpose_node_new = std::make_shared<Transpose>(reshape_node_new, order);
        for (auto& input : consumers) {
            input.replace_source_output(transpose_node_new);
        }
        return true;
    };

    auto m = std::make_shared<Matcher>(matmul, matcher_name);
    this->register_matcher(m, callback);
}

ReplaceTransposeBeforeMatMul::ReplaceTransposeBeforeMatMul() {
    MATCHER_SCOPE(ReplaceTransposeBeforeMatMul);

    auto constant_pattern = wrap_type<Constant>();
    auto fq = wrap_type<FakeQuantize>({any_input(), any_input(), any_input(), any_input(), any_input()});
    auto transpose1 = wrap_type<Transpose>({fq, any_input()});
    auto reshape1 = wrap_type<Reshape>({transpose1, constant_pattern});
    auto transpose2 = wrap_type<Transpose>({reshape1, any_input()});
    auto reshape2 = wrap_type<Reshape>({transpose2, constant_pattern});
    auto matmul = wrap_type<MatMul>({reshape2, any_input()});

    matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        // Replace transpose with reshape and transpose
        auto fq_node = pattern_map.at(fq).get_node_shared_ptr();
        auto transpose1_node = pattern_map.at(transpose1).get_node_shared_ptr();
        auto reshape1_node = pattern_map.at(reshape1).get_node_shared_ptr();
        auto transpose2_node = pattern_map.at(transpose2).get_node_shared_ptr();
        auto reshape2_node = pattern_map.at(reshape2).get_node_shared_ptr();
        auto reshape_node_new = append_reshape(fq_node, {reshape2_node->get_output_shape(0)});
        auto transpose_node_new = append_simple_transpose(reshape_node_new);
        reshape2_node->input(0).replace_source_output(transpose_node_new->output(0));
        return true;
    };

    auto m = std::make_shared<Matcher>(matmul, matcher_name);
    this->register_matcher(m, callback);
}

RemoveTransposeBeforeAdd::RemoveTransposeBeforeAdd() {
    MATCHER_SCOPE(RemoveTransposeBeforeAdd);

    auto constant_pattern = wrap_type<Constant>();
    auto fq1 = wrap_type<FakeQuantize>({any_input(), any_input(), any_input(), any_input(), any_input()});
    auto fq2 = wrap_type<FakeQuantize>({any_input(), any_input(), any_input(), any_input(), any_input()});
    auto transpose = wrap_type<Transpose>({fq1, any_input()});
    auto add = wrap_type<Add>({transpose, fq2});
    auto fq = wrap_type<FakeQuantize>({add, any_input(), any_input(), any_input(), any_input()});
    auto convolution = wrap_type<Convolution>({fq, any_input()});

    matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        // Remove redundant transpose
        auto add_node = pattern_map.at(add).get_node_shared_ptr();
        auto fq1_node = pattern_map.at(fq1).get_node_shared_ptr();
        auto transpose_node = pattern_map.at(transpose).get_node_shared_ptr();
        auto reshape_node_new = append_reshape(fq1_node, {transpose_node->get_output_shape(0)});
        add_node->input(0).replace_source_output(reshape_node_new->output(0));
        return true;
    };

    auto m = std::make_shared<Matcher>(convolution, matcher_name);
    this->register_matcher(m, callback);
}

InsertTransposeBeforeMultiply::InsertTransposeBeforeMultiply() {
    MATCHER_SCOPE(InsertTransposeBeforeMultiply);

    auto constant_pattern = wrap_type<Constant>();
    auto fq0 = wrap_type<FakeQuantize>({any_input(), any_input(), any_input(), any_input(), any_input()});
    auto reshape0 = wrap_type<Reshape>({fq0, constant_pattern});
    auto transpose = wrap_type<Transpose>({reshape0, any_input()});
    auto reshape1 = wrap_type<Reshape>({transpose, constant_pattern});
    auto sigmoid = wrap_type<Sigmoid>({reshape1});
    //auto fq1 = wrap_type<FakeQuantize>({sigmoid, any_input(), any_input(), any_input(), any_input()});
    //auto reshape2 = wrap_type<Reshape>({fq1, constant_pattern});
    //auto multiply = wrap_type<Multiply>({any_input(), reshape2});

    matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        // Add transpose
        auto fq0_node = pattern_map.at(fq0).get_node_shared_ptr();
        auto reshape1_node = pattern_map.at(reshape1).get_node_shared_ptr();
        auto sigmoid_node = pattern_map.at(sigmoid).get_node_shared_ptr();
        auto reshape_node_new = append_reshape(fq0_node, {reshape1_node->get_output_shape(0)});
        sigmoid_node->input(0).replace_source_output(reshape_node_new->output(0));
        return true;
    };

    auto m = std::make_shared<Matcher>(sigmoid, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
