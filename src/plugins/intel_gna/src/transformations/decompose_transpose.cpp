// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/decompose_transpose.hpp"

#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/utils/utils.hpp>
#include "backend/gna_limitations.hpp"


using namespace ov::opset12;
using namespace ov::intel_gna::limitations;

namespace ov {
namespace intel_gna {
namespace pass {

struct TransposeData {
    size_t N;
    size_t C;
    size_t H;
    size_t W;
    ov::Output<ov::Node> input;
    size_t input_size;
    std::vector<int32_t> order;
    std::string name;
};

static bool GetVerifiedTransposeData(const std::shared_ptr<Transpose> transpose, TransposeData& transpose_data) {
    const auto input_shape = transpose->get_input_shape(0);
    const auto input_size = input_shape.size();

    transpose_data.N = 1;
    transpose_data.C = 1;
    transpose_data.H = input_shape[input_size - 2];
    transpose_data.W = input_shape[input_size - 1];

    if (input_size == 4) {
        transpose_data.N = input_shape[0];
        transpose_data.C = input_shape[1];
    } else if (input_size == 3) {
        transpose_data.C = input_shape[0];
    }

    auto order_const = std::dynamic_pointer_cast<Constant>(transpose->input_value(1).get_node_shared_ptr());
    IE_ASSERT(order_const);

    transpose_data.input = transpose->input_value(0);
    transpose_data.input_size = input_size;
    transpose_data.order = order_const->cast_vector<int32_t>();
    transpose_data.name = transpose->get_friendly_name();

    return true;
}

static bool DecomposeTransposeType1(const std::shared_ptr<Transpose> transpose,
                                       const TransposeData& transpose_data) {
    size_t H_new = transpose_data.H;
    size_t W_new = transpose_data.W;

    if (transpose_data.order == std::vector<int32_t>{0, 3, 1, 2} ||
        transpose_data.order == std::vector<int32_t>{2, 0, 1}) {
        H_new = transpose_data.C * transpose_data.H;
    } else if (transpose_data.order == std::vector<int32_t>{0, 2, 3, 1} ||
               transpose_data.order == std::vector<int32_t>{1, 2, 0}) {
        H_new = transpose_data.C;
        W_new = transpose_data.H * transpose_data.W;
    } else if ((transpose_data.order == std::vector<int32_t>{0, 1, 3, 2} ||
               transpose_data.order == std::vector<int32_t>{0, 2, 1} ||
               transpose_data.order == std::vector<int32_t>{1, 0}) &&
               transpose_data.C == 1) {
    } else
        return false;

    const auto output_shape = transpose->output(0).get_shape();

    // GNA-compatible transpose
    if ((H_new != transpose_data.H || W_new != transpose_data.W) &&
        (H_new <= Limitations::kTransposeMaxMinDim || W_new <= Limitations::kTransposeMaxMinDim)) {
        auto reshape = std::make_shared<Reshape>(transpose_data.input,
            Constant::create(element::i32, Shape{2}, {H_new, W_new}), false);
        auto transpose_new = std::make_shared<Transpose>(reshape,
            Constant::create(element::i32, Shape{2}, {1, 0}));
        reshape = std::make_shared<Reshape>(transpose_new,
            Constant::create(element::i32, Shape{output_shape.size()}, output_shape)->output(0),
            false);
        replace_node(transpose, reshape);
        reshape->set_friendly_name(transpose_data.name);
        return true;

    // GNA-incompatible transpose
    } else if ((H_new % Limitations::kTransposeMaxMinDim) == 0) {
        // Find prime factors of W_new
        std::vector<size_t> factors;
        size_t W_tmp = W_new;
        size_t p = 2;

        while (W_tmp * W_tmp >= p * p) {
            if ((W_tmp % p) == 0) {
                factors.push_back(p);
                W_tmp = W_tmp / p;
            } else {
                p++;
            }
        }

        if (W_tmp > 1) {
            factors.push_back(W_tmp);
        }

        // Check if there are any factors too large for GNA transpose
        bool feasible = true;

        for (size_t i = 0; i < factors.size(); i++) {
            if (factors[i] > Limitations::kTransposeMaxMinDim) {
                feasible = false;
            }
        }

        // Perform feasible transformations
        if (feasible) {
            // Combine prime factors if possible
            std::vector<size_t> combined_factors;
            size_t new_factor = 1;

            for (size_t i = 0; i < factors.size(); i++) {
                size_t product = new_factor * factors[i];
                if (product > 8) {
                    combined_factors.push_back(new_factor);
                    new_factor = factors[i];
                } else {
                    new_factor = product;
                }
            }

            combined_factors.push_back(new_factor);
            // Generate transpose transformation
            OutputVector upstream;
            upstream.push_back(transpose_data.input);

            for (size_t i = 0; i < combined_factors.size(); i++) {
                auto reshape = std::make_shared<Reshape>(upstream[0],
                    Constant::create(element::i32, Shape{2}, {H_new * W_new / combined_factors[i], combined_factors[i]}), false);
                auto transpose = std::make_shared<Transpose>(reshape,
                    Constant::create(element::i32, Shape{2}, {1, 0}));
                upstream[0] = transpose;
            }

            auto reshape = std::make_shared<Reshape>(upstream[0],
                Constant::create(element::i32, Shape{output_shape.size()}, output_shape), false);
            ngraph::replace_node(transpose, reshape);
            reshape->set_friendly_name(transpose_data.name);
            return true;
        }
    }
    return false;
}

static bool DecomposeTransposeType2(const std::shared_ptr<Transpose> transpose,
                                      const TransposeData& transpose_data) {
    if (transpose_data.H <= 8 && transpose_data.W % Limitations::kTransposeMaxMinDim == 0 &&
        transpose_data.C * transpose_data.W <= Limitations::kTransposeMaxMinDim &&
        (transpose_data.order == std::vector<int32_t>{0, 2, 1, 3} ||
        transpose_data.order == std::vector<int32_t>{1, 0, 2})) {

        // We only allow decomposition of small problems to prevent creation of large number of layers
        if (transpose_data.C > 16)
            return false;

        // Split into separate HxW matrices
        size_t axis = transpose_data.input_size - 3;
        const auto axis_node = Constant::create(element::i32, Shape{}, {axis});
        const auto split = std::make_shared<Split>(transpose_data.input, axis_node, transpose_data.C);

        OutputVector chunks;

        for (size_t c = 0; c < transpose_data.C; c++) {
            auto reshape = std::make_shared<Reshape>(split->output(c),
                Constant::create(element::i32, Shape{2}, {transpose_data.H, transpose_data.W}), false);
            auto transpose_part = std::make_shared<Transpose>(reshape,
                Constant::create(element::i32, Shape{2}, {1, 0}));
            chunks.push_back(transpose_part);
        }

        auto concat = std::make_shared<Concat>(chunks, 0);
        auto transpose_final = std::make_shared<Transpose>(concat,
            Constant::create(element::i32, Shape{2}, {1, 0}));
        auto reshape = (transpose_data.input_size == 3) ? std::make_shared<Reshape>(transpose_final,
            Constant::create(element::i32, Shape{3}, {transpose_data.H, transpose_data.C, transpose_data.W}), false)
            : std::make_shared<Reshape>(transpose_final,
                Constant::create(element::i32, Shape{4}, {transpose_data.N, transpose_data.H, transpose_data.C, transpose_data.W}), false);
        ngraph::replace_node(transpose, reshape);
        reshape->set_friendly_name(transpose_data.name);
        return true;
    }

    return false;
}

static std::shared_ptr<Node> GetTransposeOrder(const TransposeData& transpose_data) {
    std::shared_ptr<Node> transpose_const;

    if (transpose_data.input_size == 2) {
        transpose_const = Constant::create(element::i32, Shape{2}, {1, 0});
    } else if (transpose_data.input_size == 3) {
        transpose_const = Constant::create(element::i32, Shape{3}, {0, 2, 1});
    } else {
        transpose_const = Constant::create(element::i32, Shape{4}, {0, 1, 3, 2});
    }
    return transpose_const;
}

static bool DecomposeTransposeType3(const std::shared_ptr<Transpose> transpose,
                                    const TransposeData& transpose_data) {
    if (transpose_data.order == std::vector<int32_t>{0, 1, 3, 2} ||
        transpose_data.order == std::vector<int32_t>{0, 2, 1} ||
        transpose_data.order == std::vector<int32_t>{1, 0}) {

        OutputVector chunks;

        // Split to 2D transpose case
        if (transpose_data.C > 1) {
            size_t axis = transpose_data.input_size - 3;
            const auto axis_node = Constant::create(element::i32, Shape{}, {axis});
            auto split = std::make_shared<Split>(transpose_data.input, axis_node, transpose_data.C);

            for (size_t c = 0; c < transpose_data.C; c++) {
                chunks.push_back(split->output(c));
            }
        } else {
            chunks.push_back(transpose_data.input);
        }

        //auto transpose_const = std::make_shared<Constant>(element::i32, Shape(transpose_data.order.size()), transpose_data.order);
        std::shared_ptr<Node> transpose_const = GetTransposeOrder(transpose_data);

        OutputVector transpose_parts;
        for (size_t c = 0; c < transpose_data.C; c++) {

            if (transpose_data.H <= 8) {
                auto new_transpose = std::make_shared<Transpose>(chunks[c], transpose_const);
                transpose_parts.push_back(new_transpose->output(0));
            } else if (transpose_data.H % Limitations::kTransposeMaxMinDim == 0 &&
                       transpose_data.W % Limitations::kTransposeMaxMinDim == 0) {
                // We only allow decomposition of small problems to prevent creation of large number of layers
                auto const total_no_of_layers = (transpose_data.H * transpose_data.W) / 64 +
                                                2 * transpose_data.H / Limitations::kTransposeMaxMinDim;

                if (total_no_of_layers > 24)
                    return false;

                // Split input into row blocks of height 8
                size_t H_div_8 = transpose_data.H / Limitations::kTransposeMaxMinDim;
                size_t W_div_8 = transpose_data.W / Limitations::kTransposeMaxMinDim;
                size_t axis = transpose_data.input_size - 2;
                const auto axis_node = Constant::create(element::i32, Shape{}, {axis});
                auto rowblock = std::make_shared<Split>(chunks[c], axis_node, H_div_8);

                // Transpose row blocks
                std::vector<OutputVector> subblock;

                for (size_t i = 0; i < H_div_8; i++) {
                    auto new_transpose = std::make_shared<Transpose>(rowblock->output(i), transpose_const);

                    // Split transposed row blocks into 8x8 blocks
                    OutputVector block;
                    auto split_rowblock = std::make_shared<Split>(new_transpose->output(0), axis_node, W_div_8);

                    for (size_t j = 0; j < W_div_8; j++) {
                        block.push_back(split_rowblock->output(j));
                    }

                    subblock.push_back(block);
                }

                // Un-transpose all 8x8 blocks
                for (size_t i = 0; i < H_div_8; i++) {
                    for (size_t j = 0; j < W_div_8; j++) {
                        auto new_transpose = std::make_shared<Transpose>(subblock[i][j], transpose_const);
                        subblock[i][j] = new_transpose->output(0);
                    }
                }

                // Concatenate into block columns
                OutputVector colblock;

                for (size_t j = 0; j < W_div_8; j++) {
                    OutputVector blocks;

                    for (size_t i = 0; i < H_div_8; i++) {
                        blocks.push_back(subblock[i][j]);
                    }

                    auto concat = std::make_shared<Concat>(blocks, transpose_data.input_size - 2);
                    colblock.push_back(concat->output(0));
                }

                // Transpose each block column
                OutputVector transposed_rowblock;
                for (size_t j = 0; j < W_div_8; j++) {
                    auto new_transpose = std::make_shared<Transpose>(colblock[j], transpose_const);
                    transposed_rowblock.push_back(new_transpose->output(0));
                }

                // Contenate to form single final transposed tensor
                if (transposed_rowblock.size() > 1) {
                    auto concat = std::make_shared<Concat>(transposed_rowblock, transpose_data.input_size - 2);
                    transpose_parts.push_back(concat->output(0));
                } else {
                    transpose_parts.push_back(transposed_rowblock[0]);
                }
            }
        }

        // Contenate parts to form final transposed tensor
        if (transpose_data.C > 1) {
            size_t axis = transpose_data.input_size - 3;
            auto concat = std::make_shared<Concat>(transpose_parts, axis);
            ngraph::replace_node(transpose, concat);
            concat->set_friendly_name(transpose_data.name);
        } else {
            ngraph::replace_node(transpose, transpose_parts[0].get_node_shared_ptr());
            transpose_parts[0].get_node_shared_ptr()->set_friendly_name(transpose_data.name);
        }
        return true;
    }

    return false;
}

static bool Convert(std::shared_ptr<Node> transpose_node) {
    const auto transpose = std::dynamic_pointer_cast<Transpose>(transpose_node);
    TransposeData transpose_data = {};

    if (!GetVerifiedTransposeData(transpose, transpose_data))
        return false;

    if (DecomposeTransposeType1(transpose, transpose_data))
        return true;

    if (DecomposeTransposeType2(transpose, transpose_data))
        return true;

    if (DecomposeTransposeType3(transpose, transpose_data))
        return true;

    THROW_GNA_EXCEPTION << "Transpose layer " << transpose_data.name << " is not supported by GNA Plugin";

    return false;
}

static std::function<bool(Output<Node>)> verify_transpose() {
    return [=](Output<Node>& output) -> bool {
        auto input_shape = output.get_node_shared_ptr()->get_input_shape(0);

        if (Limitations::is_transpose_supported(input_shape))
            return false;

        if (input_shape.size() == 4 && input_shape[0] != 1)
            return false;
    };
}

DecomposeTranspose::DecomposeTranspose() {
    MATCHER_SCOPE(DecomposeTranspose);

    auto transpose_order = ngraph::pattern::wrap_type<Constant>();
    auto transpose =
        ngraph::pattern::wrap_type<Transpose>({ngraph::pattern::any_input(), transpose_order}, verify_transpose());

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        return Convert(pattern_map.at(transpose).get_node_shared_ptr());
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
