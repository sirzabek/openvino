// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/replace_gna_nhwc_layers.hpp"

#include <vector>

#include "common/graph_utils.hpp"
#include "openvino/cc/ngraph/itt.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ops/gna_convolution.hpp"
#include "ops/gna_dwsc.hpp"
#include "ops/gna_max_pool.hpp"
#include "transformations/utils/transformation_helper.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ov::opset12;
using namespace ov::pass::pattern;
using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::pass::helper;

NGRAPH_RTTI_DEFINITION(ov::intel_gna::pass::ReplaceGnaNHWCLayers, "ReplaceGnaNHWCLayers");
NGRAPH_RTTI_DEFINITION(ov::intel_gna::pass::SubstituteGNAConvolution, "SubstituteGNAConvolution");
NGRAPH_RTTI_DEFINITION(ov::intel_gna::pass::SubstituteGNADwsc, "SubstituteGNADwsc");
NGRAPH_RTTI_DEFINITION(ov::intel_gna::pass::SubstituteGNAMaxPool, "SubstituteGNAMaxPool");

namespace {
ov::Shape make_transpose_order_nchw2nhwc(size_t shape_size);
ov::Shape make_transpose_order_nhwc2nchw(size_t shape_size);

/* transpose orders
   before convolution layout conversion NCHW -> NHWC
    3D: NCX {0, 1, 2} -> NXC {0, 2, 1}
    4D: NCHW {0, 1, 2, 3} -> NHWC {0, 2, 3, 1}
   after convolution layout conversion NHWC -> NCHW
   3D: NXC {0, 1, 2} -> NCX {0, 2, 1}
   4D: NHWC {0, 1, 2, 3} -> NCHW {0, 3, 1, 2}
   so just
   1) temp = A[N - 1]
   2) move A[j] -> A[j + 1] for 1 <= j <= N - 2
   3) A[1] = temp
*/

ov::Shape make_transpose_order_nchw2nhwc(size_t shape_size) {
    ov::Shape shape(shape_size);
    std::iota(shape.begin(), shape.end(), 0);

    for (size_t i = 1; i < shape.size() - 1; ++i)
        shape[i] = shape[i + 1];

    *(shape.end() - 1) = 1;

    return shape;
}

ov::Shape make_transpose_order_nhwc2nchw(size_t shape_size) {
    ov::Shape shape(shape_size);
    std::iota(shape.begin(), shape.end(), 0);

    const size_t channels_position = *(shape.end() - 1);

    for (size_t i = shape.size() - 1; i > 0; --i)
        shape[i] = shape[i - 1];

    shape[1] = channels_position;

    return shape;
}

}  // namespace

namespace SubstituteGNAConvolutionNS {

template <typename From, typename To>
bool do_transformation(std::shared_ptr<ov::Node> convolution);

template<typename From, typename To>
bool do_transformation(std::shared_ptr<ov::Node> convolution) {
    auto convolution_node = std::dynamic_pointer_cast<From>(convolution);
    auto convolution_input_data_node = convolution_node->input_value(0);
    auto convolution_filters_node = convolution_node->input_value(1);
    const auto convolution_input_shape_size = convolution_node->get_input_shape(0).size();
    const auto convolution_filters_shape_size = convolution_node->get_input_shape(1).size();

    if (convolution_input_shape_size != 3 && convolution_input_shape_size != 4) {
        std::cout << "ReplaceGnaNHWCLayers: unsupported convolution shape size " << convolution_input_shape_size
                  << std::endl;
        return false;
    }

    const ov::Shape transpose_before_order = make_transpose_order_nchw2nhwc(convolution_input_shape_size);

    auto transpose_const =
        Constant::create(element::i32, ov::Shape{transpose_before_order.size()}, transpose_before_order);
    auto transpose_const_filters = transpose_const;
    auto transpose_filters_input = convolution_filters_node;

    // We need to reshape the filters if we're dealing with GroupConvolution
    if (convolution_filters_shape_size > convolution_input_shape_size) {
        ov::Shape reshape_before_order = convolution_node->get_input_shape(1);
        reshape_before_order.erase(reshape_before_order.begin() + 2);
        const auto reshape_const_filters =
            Constant::create(element::i32, ov::Shape{reshape_before_order.size()}, reshape_before_order);
        ov::Shape transpose_before_order = ov::Shape{1, 2, 3, 0};
        transpose_const_filters =
            Constant::create(element::i32, ov::Shape{transpose_before_order.size()}, transpose_before_order);
        transpose_filters_input = std::make_shared<Reshape>(convolution_filters_node, reshape_const_filters, false);
    }

    auto transpose_before = std::make_shared<Transpose>(convolution_input_data_node, transpose_const);

    auto transpose_conv_constant = std::make_shared<Transpose>(transpose_filters_input, transpose_const_filters);
    auto conv_new = std::make_shared<To>(transpose_before,
                                         transpose_conv_constant,
                                         convolution_node->get_strides(),
                                         convolution_node->get_pads_begin(),
                                         convolution_node->get_pads_end(),
                                         convolution_node->get_dilations(),
                                         convolution_node->get_auto_pad());

    const ov::Shape transpose_after_order = make_transpose_order_nhwc2nchw(conv_new->get_output_shape(0).size());

    auto transpose_after = std::make_shared<Transpose>(
        conv_new,
        Constant::create(element::i32, ov::Shape{transpose_after_order.size()}, transpose_after_order));

    ov::copy_runtime_info(convolution_node,
                          {transpose_before, transpose_const, conv_new, transpose_after, transpose_conv_constant});

    ov::replace_output_update_name(convolution->output(0), transpose_after->output(0));

    return true;
}

}  // namespace SubstituteGNAConvolutionNS

namespace SubstituteGNAMaxPoolNS {

bool do_transformation(std::shared_ptr<ov::Node> convolution);

bool do_transformation(std::shared_ptr<ov::Node> max_pool) {
    auto max_pool_node = std::dynamic_pointer_cast<ov::op::v1::MaxPool>(max_pool);
    auto max_pool_input_data_node = max_pool_node->input_value(0);
    const ov::Shape max_pool_input_shape = max_pool_node->get_input_shape(0);

    const ov::Shape transpose_before_order = make_transpose_order_nchw2nhwc(max_pool_input_shape.size());

    auto transpose_const =
        Constant::create(element::i32, ov::Shape{transpose_before_order.size()}, transpose_before_order);

    auto transpose_before = std::make_shared<Transpose>(max_pool_input_data_node, transpose_const);

    auto max_pool_new = std::make_shared<ov::intel_gna::op::GNAMaxPool>(transpose_before,
                                                                        max_pool_node->get_strides(),
                                                                        max_pool_node->get_pads_begin(),
                                                                        max_pool_node->get_pads_end(),
                                                                        max_pool_node->get_kernel(),
                                                                        max_pool_node->get_rounding_type(),
                                                                        max_pool_node->get_auto_pad());

    const ov::Shape transpose_after_order = make_transpose_order_nhwc2nchw(max_pool_new->get_output_shape(0).size());

    auto transpose_after = std::make_shared<Transpose>(
        max_pool_new,
        Constant::create(element::i32, ov::Shape{transpose_after_order.size()}, transpose_after_order));

    ov::copy_runtime_info(max_pool_node, {transpose_before, transpose_const, max_pool_new, transpose_after});

    ov::replace_output_update_name(max_pool->output(0), transpose_after->output(0));

    return true;
}

}  // namespace SubstituteGNAMaxPoolNS

// ----------------------------------------------------------------------------

ov::intel_gna::pass::SubstituteGNAConvolution::SubstituteGNAConvolution() {
    MATCHER_SCOPE(SubstituteGNAConvolution);

    auto convolution = wrap_type<Convolution>();

    matcher_pass_callback callback = [=](Matcher& m) {
        auto convolution_node = std::dynamic_pointer_cast<Convolution>(m.get_match_root());
        if (!convolution_node) {
            return false;
        }

        return SubstituteGNAConvolutionNS::do_transformation<Convolution, ov::intel_gna::op::GNAConvolution>(
            convolution_node);
    };

    auto m = std::make_shared<Matcher>(convolution, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_gna::pass::SubstituteGNADwsc::SubstituteGNADwsc() {
    MATCHER_SCOPE(SubstituteGNADwsc);

    auto convolution = wrap_type<GroupConvolution>();

    matcher_pass_callback callback = [=](Matcher& m) {
        auto convolution_node = std::dynamic_pointer_cast<GroupConvolution>(m.get_match_root());
        if (!convolution_node) {
            return false;
        }

        return SubstituteGNAConvolutionNS::do_transformation<GroupConvolution, ov::intel_gna::op::GNADwsc>(
            convolution_node);
    };

    auto m = std::make_shared<Matcher>(convolution, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_gna::pass::SubstituteGNAMaxPool::SubstituteGNAMaxPool() {
    MATCHER_SCOPE(SubstituteGNAMaxPool);

    auto max_pool = wrap_type<ov::op::v1::MaxPool>();

    matcher_pass_callback callback = [=](Matcher& m) {
        auto max_pool_node = std::dynamic_pointer_cast<ov::op::v1::MaxPool>(m.get_match_root());
        if (!max_pool_node) {
            return false;
        }

        return SubstituteGNAMaxPoolNS::do_transformation(max_pool_node);
    };

    auto m = std::make_shared<Matcher>(max_pool, matcher_name);
    this->register_matcher(m, callback);
}

bool ov::intel_gna::pass::ReplaceGnaNHWCLayers::run_on_model(const std::shared_ptr<Model>& function) {
    RUN_ON_MODEL_SCOPE(ReplaceGnaNHWCLayers);

    ov::pass::Manager manager(get_pass_config());
    manager.register_pass<ov::intel_gna::pass::SubstituteGNAConvolution>();
    manager.register_pass<ov::intel_gna::pass::SubstituteGNADwsc>();
    manager.register_pass<ov::intel_gna::pass::SubstituteGNAMaxPool>();
    manager.run_passes(function);

    return false;
}
