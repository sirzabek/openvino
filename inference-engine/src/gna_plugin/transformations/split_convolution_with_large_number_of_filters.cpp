// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/split_convolution_with_large_number_of_filters.hpp"

#include <numeric>

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include "backend/gna_limitations.hpp"
#include "layers/gna_split_layer.hpp"
#include "layers/gna_convolution_layer.hpp"
#include "utils/transformation_helper.hpp"

using namespace GNAPluginNS;

NGRAPH_RTTI_DEFINITION(SplitConvolutionFilter, "SplitConvolutionFilter", 0);
NGRAPH_RTTI_DEFINITION(SplitConvolutionFilterWithBias, "SplitConvolutionFilterWithBias", 0);
NGRAPH_RTTI_DEFINITION(SplitConvolutionFilterWithFq, "SplitConvolutionFilterWithFq", 0);

static std::shared_ptr<ngraph::Node> getConvForMatcher() {
    return ngraph::pattern::wrap_type<ngraph::opset8::Convolution>({ ngraph::pattern::any_input(),
    ngraph::pattern::any_input() }, [](const ngraph::Output<ngraph::Node>& node) {
            auto conv = dynamic_cast<ngraph::opset8::Convolution*>(node.get_node());
            IE_ASSERT(conv != nullptr);
            auto& num_of_filters = conv->get_input_shape(1).at(0);

            // Split only convolutions with large number of filters
            if (num_of_filters <= GNALimitations::convMaxFiltersNumGna3_0) {
                return false;
            }

            // Split only 1D convolutions
            if (conv->get_input_shape(0)[2] != 1) {
                return false;
            }

            return true;
        });
}

static bool Convert(std::shared_ptr<ngraph::Node> conv,
                    std::shared_ptr<ngraph::Node> add,
                    std::shared_ptr<ngraph::Node> bias,
                    std::shared_ptr<ngraph::Node> fq) {
    auto& num_of_filters = conv->get_input_shape(1).at(0);
    auto split_filter_sizes = GetAlignedSplitSizes(num_of_filters, GNALimitations::convMaxFiltersNumGna3_0, 1);
    IE_ASSERT(split_filter_sizes.size() > 1);

    const auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape({split_filter_sizes.size()}), split_filter_sizes);

    ngraph::OutputVector split_bias;

    if (bias) {
        const auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});
        const auto const_split = std::make_shared<ngraph::opset8::VariadicSplit>(bias, axis_node, sizes_node);
        split_bias = const_split->outputs();
    }

    const auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
    const auto conv_split = std::make_shared<ngraph::opset8::VariadicSplit>(conv->input_value(1).get_node_shared_ptr(), axis_node, sizes_node);
    const auto split_conv_count = split_filter_sizes.size();
    ngraph::OutputVector split_filters = conv_split->outputs();
    ngraph::OutputVector conv_outputs;
    std::shared_ptr<ngraph::Node> root_node = fq ? fq : (add ? add : conv);
    std::shared_ptr<ngraph::Node> concat;

    for (int i = 0; i < split_conv_count; i++) {
        std::shared_ptr<ngraph::Node> output = conv->clone_with_new_inputs({conv->input_value(0), split_filters.at(i)});
        ngraph::copy_runtime_info(conv, output);
        output->set_friendly_name(conv->get_friendly_name() + "_" + std::to_string(i));

        if (bias) {
            output = add->clone_with_new_inputs({output, split_bias.at(i)});
            ngraph::copy_runtime_info(add, output);
            output->set_friendly_name(add->get_friendly_name() + "_" + std::to_string(i));
        }

        if (fq) {
            output = fq->clone_with_new_inputs({output, fq->input_value(1), fq->input_value(2),
                fq->input_value(3), fq->input_value(4)});
            ngraph::copy_runtime_info(fq, output);
        }

        conv_outputs.push_back(output);
    }

    // If convolution output is not 1x1 following concat needs to be decomposed
    const auto conv_output_width = conv_outputs.at(0).get_shape()[3];

    if (conv_output_width != 1) {
        ngraph::OutputVector conv_output_splits;

        for (size_t i = 0; i < split_conv_count; i++) {
            const auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {3});
            const auto split = std::make_shared<ngraph::opset8::Split>(conv_outputs.at(i), axis_node, conv_output_width);
            conv_output_splits.push_back(split);
        }

        ngraph::OutputVector concat_chunks;
        for (size_t c = 0; c < conv_output_width; c++) {
            ngraph::OutputVector concat_sub_chunks;

            for (size_t i = 0; i < split_conv_count; i++) {
                concat_sub_chunks.push_back(conv_output_splits[i].get_node()->output(c));
            }

            auto new_sub_concat = std::make_shared<ngraph::opset8::Concat>(concat_sub_chunks, 1);
            concat_chunks.push_back(new_sub_concat->output(0));
        }
        concat = std::make_shared<ngraph::opset8::Concat>(concat_chunks, 3);
    } else {
        concat = std::make_shared<ngraph::opset8::Concat>(conv_outputs, 1);
    }

    ngraph::copy_runtime_info(root_node, concat);
    concat->set_friendly_name(root_node->get_friendly_name());
    ngraph::replace_node(root_node, concat);
    return true;
}

SplitConvolutionFilter::SplitConvolutionFilter() {
    MATCHER_SCOPE(SplitConvolutionFilter);
    auto conv = getConvForMatcher();
    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        const auto& pattern_map = m.get_pattern_value_map();
        return Convert(pattern_map.at(conv).get_node_shared_ptr(), nullptr, nullptr, nullptr);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(conv, matcher_name);
    this->register_matcher(m, callback);
}

SplitConvolutionFilterWithBias::SplitConvolutionFilterWithBias() {
    MATCHER_SCOPE(SplitConvolutionFilterWithBias);
    auto conv = getConvForMatcher();
    auto bias = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto add = ngraph::pattern::wrap_type<ngraph::opset8::Add>({conv, bias});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto conv_node = pattern_map.at(conv).get_node_shared_ptr();
        auto add_node = pattern_map.at(add).get_node_shared_ptr();
        std::shared_ptr<ngraph::Node> bias_node = nullptr;

        if (add_node && (bias_node = VerifyBiasGetConst(conv_node, add_node)) == nullptr)
            return false;

        return Convert(conv_node, add_node, bias_node, nullptr);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(add, matcher_name);
    this->register_matcher(m, callback);
}

SplitConvolutionFilterWithFq::SplitConvolutionFilterWithFq() {
    MATCHER_SCOPE(SplitConvolutionFilterWithFq);
    auto conv = getConvForMatcher();
    auto bias = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto add = ngraph::pattern::wrap_type<ngraph::opset8::Add>({conv, bias});
    auto conv_output = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{conv, add});
    auto out_fq = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({conv_output,
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>()});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto conv_node = pattern_map.at(conv).get_node_shared_ptr();
        auto add_it = pattern_map.find(add);
        auto add_node = (add_it == std::end(pattern_map) ? nullptr : add_it->second.get_node_shared_ptr());
        std::shared_ptr<ngraph::Node> bias_node = nullptr;

        if (add_node && (bias_node = VerifyBiasGetConst(conv_node, add_node)) == nullptr)
            return false;

        return Convert(conv_node, add_node, bias_node, pattern_map.at(out_fq).get_node_shared_ptr());
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(out_fq, matcher_name);
    this->register_matcher(m, callback);
}
