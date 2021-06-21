// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_padded2valid_conv.hpp"

#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ie_common.h>


using namespace GNAPluginNS;
using namespace ngraph;
using namespace op;

namespace {
struct GraphData {
    std::shared_ptr<opset1::Convolution> conv;
    std::shared_ptr<Transpose> leading_transpose;
    std::shared_ptr<Transpose> trailing_transpose;
    std::shared_ptr<opset1::MaxPool> max_pool;
    std::shared_ptr<op::util::UnaryElementwiseArithmetic> af;
    std::shared_ptr<Node> bias_const;
    std::shared_ptr<Node>last_op_in_sequence_for_replacement;
};

struct ConvData {
    size_t input_height;
    size_t input_width;
    size_t input_channel_count;
    size_t filter_height;
    size_t filter_width;
    size_t filter_count;
    size_t filter_dilation_width;
    size_t filter_dilation_height;
    size_t filter_stride_width;
    size_t filter_stride_height;
    size_t pads_begin_width;
    size_t pads_begin_height;
    size_t pads_end_width;
    size_t pads_end_height;
    op::PadType padding_type;
    size_t output_channel_count;
    Shape output_shape;
    element::Type element_type;
};

bool TransposeOrderMatches(std::shared_ptr<Transpose> transpose, std::vector<int64_t> order) {
    if (!transpose)
        return false;
    const Output<Node>& transpose_order = transpose->input_value(1);
    auto transpose_order_dim = transpose_order.get_shape().size();

    if (transpose_order_dim != 1 || transpose_order.get_shape()[0] != order.size())
        return false;

    auto const_with_order_values = std::dynamic_pointer_cast<opset1::Constant>(transpose_order.get_node_shared_ptr());
    if (!const_with_order_values)
        return false;

    const int64_t* data = const_with_order_values->get_data_ptr<int64_t>();
    if (!data)
        return false;

    for (size_t i = 0; i < order.size(); i++) {
        if (order[i] != data[i])
            return false;
    }

    return true;
}

std::shared_ptr<opset1::StridedSlice> FlatCrop(Output<Node> input, size_t offset, size_t size) {
    return std::make_shared<opset1::StridedSlice>(
        input, // data
        opset1::Constant::create(element::i64, Shape{ 2 }, { (size_t)0, offset }), // begin sice index
        opset1::Constant::create(element::i64, Shape{ 2 }, { (size_t)0, offset + size }), // end slice index
        opset1::Constant::create(element::i64, Shape{ 2 }, { (size_t)1, (size_t)1 }), // strides
        std::vector<int64_t>{1, 0},  // begin mask
        std::vector<int64_t>{1, 0}); // end mask
}

template<class T>
bool VerifyLayer(std::shared_ptr<T> layer) {
    auto layer_output_0 = layer->get_output_target_inputs(0);
    return layer_output_0.size() == 1 ? true : false;
}

template<>
bool VerifyLayer<>(std::shared_ptr<opset1::MaxPool> max_pool) {
    auto layer_output_0 = max_pool->get_output_target_inputs(0);
    if (layer_output_0.size() != 1)
        return false;

    auto pool_strides = max_pool->get_strides();
    auto pool_kernel = max_pool->get_kernel();

    // Check if MaxPool vertical stride == pool size
    // (TODO: remove when 50386 and 50379 are fixed and also verify pool_kernel[0] > 8 limitation below, gna_limitations can be used then)
    // Check if padding is VALID
    if (max_pool->get_auto_pad() != PadType::VALID ||
        pool_kernel.size() != 2 || pool_strides.size() != 2 ||
        pool_kernel[0] != pool_strides[0] || pool_kernel[0] > 8)
        return false;

    return true;
}

bool VerifyBias(std::shared_ptr<opset1::Add> conv_bias, const size_t filter_count) {
    auto add_const = std::dynamic_pointer_cast<op::Constant>(conv_bias->input_value(1).get_node_shared_ptr());

    // The add may be a normal add not bias, then we just go further
    if (add_const && shape_size(add_const->get_shape()) == filter_count)
        return true;

    // Bias size doesn't match (or dynamic bias), can't convert such convolution
    return false;
}

std::shared_ptr<opset1::Convolution> DetectVerifyConvolution(std::shared_ptr<Node> node) {
    auto conv = std::dynamic_pointer_cast<opset1::Convolution>(node);

    if (conv) {
        // check if convolution output port is connected with only one Op
        if (!VerifyLayer(node))
            return nullptr;

        const auto& input = conv->input_value(0);
        const auto& filters = conv->input_value(1);
        const auto& output_shape = conv->get_output_shape(0);

        if (!std::dynamic_pointer_cast<opset1::Constant>(filters.get_node_shared_ptr()))
            return nullptr;

        // we support only 2D conv batch 1
        if (input.get_shape().size() != 4 ||
            filters.get_shape().size() != 4 ||
            output_shape.size() != 4 ||
            conv->get_dilations().size() != 2 ||
            conv->get_strides().size() != 2 ||
            input.get_shape()[0] != 1) {
            return nullptr;
        }
    }
    return conv;
}

void FillConvData(std::shared_ptr<opset1::Convolution> conv, ConvData& conv_data) {
    conv_data.output_shape = conv->get_output_shape(0);
    conv_data.padding_type = conv->get_auto_pad();
    conv_data.input_height = conv->input_value(0).get_shape()[2];
    conv_data.input_width = conv->input_value(0).get_shape()[3];
    conv_data.input_channel_count = conv->input_value(0).get_shape()[1];
    conv_data.filter_height = conv->input_value(1).get_shape()[2];
    conv_data.filter_width = conv->input_value(1).get_shape()[3];
    conv_data.filter_count = conv->input_value(1).get_shape()[0];
    conv_data.filter_dilation_width = conv->get_dilations()[1];
    conv_data.filter_dilation_height = conv->get_dilations()[0];
    conv_data.filter_stride_width = conv->get_strides()[1];
    conv_data.filter_stride_height = conv->get_strides()[0];
    conv_data.pads_begin_height = conv->get_pads_begin()[0];
    conv_data.pads_begin_width = conv->get_pads_begin()[1];
    conv_data.pads_end_height = conv->get_pads_end()[0];
    conv_data.pads_end_width = conv->get_pads_end()[1];
    conv_data.output_channel_count = conv_data.filter_count;
    conv_data.element_type = conv->get_element_type();
}

std::shared_ptr<Transpose> DetectVerifyLeadingTranspose(std::shared_ptr<opset1::Convolution> conv) {
    const Output<Node>& input = conv->input_value(0);
    auto leading_transpose = std::dynamic_pointer_cast<Transpose>(input.get_node_shared_ptr());

    if (!leading_transpose || !TransposeOrderMatches(leading_transpose, { 0, 3, 1, 2 }))
        return nullptr;

    return leading_transpose;
}

template<class T>
std::shared_ptr<T> DetectNextLayer(std::shared_ptr<Node> node) {
    auto output_0_node = node->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
    return std::dynamic_pointer_cast<T>(output_0_node);
}

template<class T>
bool DetectOptionalLayer(GraphData& graph_data, std::shared_ptr<T> layer) {
    if ((layer = DetectNextLayer<T>(graph_data.last_op_in_sequence_for_replacement))) {
        if (!VerifyLayer(layer))
            return false;

        // disable_nhwc_to_nchw option case
        if (graph_data.trailing_transpose) {
            graph_data.last_op_in_sequence_for_replacement = layer;
        } else {
            if ((graph_data.trailing_transpose = DetectNextLayer<Transpose>(layer))) {
                graph_data.last_op_in_sequence_for_replacement = graph_data.trailing_transpose;
            } else {
                graph_data.last_op_in_sequence_for_replacement = layer;
            }
        }
    }
    return true;
}

bool DetectGraphSequence(GraphData& graph_data, const ConvData& conv_data) {
    std::shared_ptr<opset1::Add> conv_bias;
    graph_data.last_op_in_sequence_for_replacement = graph_data.conv;

    if ((graph_data.trailing_transpose = DetectNextLayer<Transpose>(graph_data.conv))) {
        graph_data.last_op_in_sequence_for_replacement = graph_data.trailing_transpose;

        if (VerifyLayer(graph_data.trailing_transpose) &&
            (conv_bias = DetectNextLayer<opset1::Add>(graph_data.trailing_transpose)) &&
            (VerifyBias(conv_bias, conv_data.filter_count))) {
            graph_data.last_op_in_sequence_for_replacement = conv_bias;
        }
    } else if ((conv_bias = DetectNextLayer<opset1::Add>(graph_data.conv))) {
        if (!VerifyLayer(conv_bias) || !(VerifyBias(conv_bias, conv_data.filter_count)))
            return false;

        if ((graph_data.trailing_transpose = DetectNextLayer<Transpose>(conv_bias))) {
            graph_data.last_op_in_sequence_for_replacement = graph_data.trailing_transpose;
        } else {
            graph_data.last_op_in_sequence_for_replacement = conv_bias;
        }
    } else {
        // TODO: should we want to support also Transpose(NHWC->NCHW) = > conv = > MaxPool => Transpose(NCHW->NHWC)
        // we need to remove continue then
        return false;
    }

    // max pooling
    if (!DetectOptionalLayer<opset1::MaxPool>(graph_data, graph_data.max_pool))
        return false;

    // and finally activation function
    if (!DetectOptionalLayer<op::util::UnaryElementwiseArithmetic>(graph_data, graph_data.af))
        return false;

    if (!graph_data.trailing_transpose || !graph_data.last_op_in_sequence_for_replacement ||
        !TransposeOrderMatches(graph_data.trailing_transpose, { 0, 2, 3, 1 }))
        return false;
    return true;
}

int32_t GetRequiredInputPadding(size_t input_size, size_t filter_size, size_t stride_size, size_t dilation_size, size_t output_size) {
    int32_t padding_size = (output_size - 1) * stride_size + (filter_size - 1) * dilation_size + 1 - input_size;

    return padding_size >= 0 ? padding_size : 0;
}

size_t CalculateOutputSize(size_t input_size, size_t filter_size, size_t stride_size, size_t dilation_size, size_t padding_size) {
    return (input_size + padding_size - ((filter_size - 1) * dilation_size + 1)) / stride_size + 1;
}

bool CalculatePadding(const GraphData& graph_data, ConvData& conv_data) {
    size_t output_channel_count = conv_data.filter_count;
    size_t output_height{ 0 };
    size_t output_width{ 0 };

    switch (conv_data.padding_type) {
    case op::PadType::EXPLICIT:
        // all paddings already set
        break;
    case op::PadType::VALID:
        conv_data.pads_begin_height = 0;
        conv_data.pads_begin_width = 0;
        conv_data.pads_end_height = 0;
        conv_data.pads_end_width = 0;
        break;
    case op::PadType::SAME_LOWER:
    case op::PadType::SAME_UPPER:
    {
        output_height = conv_data.output_shape[2];
        output_width = conv_data.output_shape[3];

        int32_t pads_width = GetRequiredInputPadding(conv_data.input_width, conv_data.filter_width,
            conv_data.filter_stride_width, conv_data.filter_dilation_width, output_width);
        int32_t pads_height = GetRequiredInputPadding(conv_data.input_height, conv_data.filter_height,
            conv_data.filter_stride_height, conv_data.filter_dilation_height, output_height);

        conv_data.pads_begin_width = conv_data.pads_end_width = pads_width / 2;
        conv_data.pads_begin_height = conv_data.pads_end_height = pads_height / 2;

        if (conv_data.padding_type == op::PadType::SAME_LOWER) {
            conv_data.pads_begin_width += (pads_width % 2);
            conv_data.pads_begin_height += (pads_height % 2);
        } else {
            conv_data.pads_end_width += (pads_width % 2);
            conv_data.pads_end_height += (pads_height % 2);
        }
        break;
    }
    default:
        break;
    }

    output_width = CalculateOutputSize(conv_data.input_width, conv_data.filter_width, conv_data.filter_stride_width,
        conv_data.filter_dilation_width, conv_data.pads_begin_width + conv_data.pads_end_width);
    output_height = CalculateOutputSize(conv_data.input_height, conv_data.filter_height, conv_data.filter_stride_height,
        conv_data.filter_dilation_height, conv_data.pads_begin_height + conv_data.pads_end_height);

    IE_ASSERT(output_channel_count == conv_data.output_shape[1]);
    IE_ASSERT(output_width == conv_data.output_shape[3]);
    IE_ASSERT(output_height == conv_data.output_shape[2]);

    // Check if any calculated padding is non-zero, otherwise there is no need to decompose such convolution
    return conv_data.pads_begin_height || conv_data.pads_end_height || conv_data.pads_begin_width || conv_data.pads_end_width;
}

void InsertPadding(OutputVector& input_rows_to_concat, size_t size, const std::shared_ptr<opset1::Convolution>& conv,
    const std::shared_ptr<opset1::Constant> padding_const, size_t biggest_padding) {

    if (size == biggest_padding) {
        input_rows_to_concat.push_back(padding_const);
    } else {
        auto slice = FlatCrop(padding_const, 0, size);
        copy_runtime_info(conv, slice);
        input_rows_to_concat.push_back(slice);
    }
}

std::shared_ptr<Node> CreatePaddedNet(const GraphData& graph_data, const ConvData& conv_data) {
    size_t flat_left_padding = conv_data.input_channel_count * conv_data.pads_begin_width;
    size_t flat_right_padding = conv_data.input_channel_count * conv_data.pads_end_width;
    size_t padded_row_size = flat_left_padding + conv_data.input_channel_count * conv_data.input_width + flat_right_padding;
    size_t flat_top_padding = padded_row_size * conv_data.pads_begin_height;
    size_t flat_bottom_padding = padded_row_size * conv_data.pads_end_height;
    size_t biggest_padding = std::max(std::max(flat_left_padding, flat_right_padding), std::max(flat_top_padding, flat_bottom_padding));

    if (conv_data.input_height > 1 && (flat_top_padding > 1 || flat_bottom_padding > 1)) {
        biggest_padding = biggest_padding > padded_row_size ? biggest_padding : padded_row_size;
    }

    auto flat_input = std::make_shared<opset1::Reshape>(graph_data.leading_transpose->input_value(0),
        op::Constant::create(element::i64, Shape{ 2 }, Shape{ 1ull, shape_size(graph_data.leading_transpose->input_value(0).get_shape()) }), false);

    // zero padding
    auto const_holding_padding = std::make_shared<opset1::Constant>(conv_data.element_type, Shape{ 1, biggest_padding }, 0);

    copy_runtime_info(graph_data.conv, const_holding_padding);
    std::shared_ptr<Node> original_row = flat_input;
    OutputVector input_rows_to_concat;

    // Add top padding
    for (size_t p = 0; p < conv_data.pads_begin_height; p++) {
        InsertPadding(input_rows_to_concat, padded_row_size, graph_data.conv, const_holding_padding, biggest_padding);
    }

    if (flat_left_padding || flat_right_padding) {
        // Pad every row of input plain if neccessary
        for (size_t h = 0; h < conv_data.input_height; h++) {
            // left padding     input     right padding
            //     |              |           |
            //     +--------------+-----------+
            //                    |
            //                 concat

            if (conv_data.input_height > 1)
                original_row = FlatCrop(flat_input, h * conv_data.input_width * conv_data.input_channel_count,
                    conv_data.input_width * conv_data.input_channel_count);
            copy_runtime_info(graph_data.conv, original_row);

            OutputVector single_row_concat_inputs;
            if (flat_left_padding) {
                InsertPadding(single_row_concat_inputs, flat_left_padding, graph_data.conv, const_holding_padding, biggest_padding);
            }
            single_row_concat_inputs.push_back(original_row);
            if (flat_right_padding) {
                InsertPadding(single_row_concat_inputs, flat_right_padding, graph_data.conv, const_holding_padding, biggest_padding);
            }
            auto padded_row_concat = std::make_shared<opset1::Concat>(single_row_concat_inputs, 1);
            copy_runtime_info(graph_data.conv, padded_row_concat);
            input_rows_to_concat.push_back(padded_row_concat);
        }
    } else {
        copy_runtime_info(graph_data.conv, original_row);
        input_rows_to_concat.push_back(original_row);
    }

    // Bottom padding
    for (size_t p = 0; p < conv_data.pads_end_height; p++) {
        InsertPadding(input_rows_to_concat, padded_row_size, graph_data.conv, const_holding_padding, biggest_padding);
    }

    auto padded_input_plane = std::make_shared<opset1::Concat>(input_rows_to_concat, 1);
    copy_runtime_info(graph_data.conv, padded_input_plane);
    return padded_input_plane;
}

void GeneratePadding(const GraphData& graph_data, const ConvData& conv_data) {
    // Add padding where neccessary

    // padding
    // padding
    // ... row ...
    // ... row ...
    // ...........
    // ... row ...
    // padding
    // padding
    auto padded_input_plane = CreatePaddedNet(graph_data, conv_data);

    auto padded_input_plane_reshaped = std::make_shared<opset1::Reshape>(padded_input_plane,
        op::Constant::create(element::i64, Shape{ 4 }, { static_cast<size_t>(1),
            conv_data.pads_begin_height + conv_data.input_height + conv_data.pads_end_height,
            conv_data.pads_begin_width + conv_data.input_width + conv_data.pads_end_width,
            conv_data.input_channel_count }), false);

    //NHWC => NCHW
    auto transposed2chw = std::make_shared<op::Transpose>(padded_input_plane_reshaped,
        op::Constant::create(element::i64, Shape{ 4 }, { 0ull, 3ull, 1ull, 2ull })->output(0));

    auto conv_copy = std::make_shared<opset1::Convolution>(
        transposed2chw->output(0),
        graph_data.conv->input_value(1),
        graph_data.conv->get_strides(),
        CoordinateDiff{ 0, 0 },
        CoordinateDiff{ 0, 0 },
        graph_data.conv->get_dilations(),
        PadType::EXPLICIT);

    replace_node(graph_data.conv, conv_copy);
}
} // namespace

// Supported cases:
//   - Transpose(NHWC->NCHW) => Conv => Transpose(NCHW->NHWC)
//   - Transpose(NHWC->NCHW) => Conv => Broadcasted add (bias) => Transpose(NCHW->NHWC)
//   - Transpose(NHWC->NCHW) => Conv => Broadcasted add (bias) => MaxPooling => Transpose(NCHW->NHWC) (2d max pool case)
//   - Transpose(NHWC->NCHW) => Conv => Broadcasted add (bias) => ActivationFunction => Transpose(NCHW->NHWC)
//   - Transpose(NHWC->NCHW) => Conv => Broadcasted add (bias) => MaxPool => ActivationFunction => Transpose(NCHW->NHWC)
//   - Transpose(NHWC->NCHW) => Conv => Transpose(NCHW->NHWC) => Bias (output of MO --disable_nhwc_to_nchw option)
//   - Transpose(NHWC->NCHW) => Conv => Transpose(NCHW->NHWC) => Bias => ActivationFunction (output of MO --disable_nhwc_to_nchw option)

NGRAPH_RTTI_DEFINITION(ConvertPadded2ValidConv, "ConvertPadded2ValidConv", 0);
bool ConvertPadded2ValidConv::run_on_function(std::shared_ptr<Function> f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;

    for (auto& node : f->get_ordered_ops()) {
        GraphData graph_data;
        ConvData conv_data;

        if ((graph_data.conv = DetectVerifyConvolution(node)) == nullptr)
            continue;

        FillConvData(graph_data.conv, conv_data);

        // We are looking for Transpose(NHWC->NCHW) => Conv => Transpose(NCHW->NHWC)
        // or similar cases so required network must be in NHWC order like in TF
        if (!(graph_data.leading_transpose = DetectVerifyLeadingTranspose(graph_data.conv)))
            continue;

        if (!DetectGraphSequence(graph_data, conv_data))
            continue;

        if (!CalculatePadding(graph_data, conv_data))
            continue;

        GeneratePadding(graph_data, conv_data);

        is_graph_modfied = true;
    }
    return is_graph_modfied;
}
