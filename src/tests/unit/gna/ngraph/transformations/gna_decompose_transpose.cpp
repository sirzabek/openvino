// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <tuple>

#include "transformations/decompose_transpose.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/opsets/opset9.hpp>
#include "backend/gna_limitations.hpp"
#include "transformations/serialize.hpp"
using namespace GNAPluginNS;
using namespace ngraph;

namespace decomposeTranspose {

struct TransposeData {
    size_t N;
    size_t C;
    size_t H;
    size_t W;
    ov::Output<ov::Node> input;
    size_t input_size;
    Shape input_shape;
    Shape order;
};

std::shared_ptr<Function> create_function(const Shape& input_shape,
                                                  const Shape& transpose_order) {
    auto input_params = std::make_shared<opset9::Parameter>(element::i32, input_shape);
    auto transpose_params = std::make_shared<opset9::Constant>(element::i32,
                                                                       Shape{transpose_order.size()},
                                                                       transpose_order);
    auto transpose = std::make_shared<opset9::Transpose>(input_params, transpose_params);
    auto result = std::make_shared<opset9::Result>(transpose);

    return std::make_shared<Function>(ResultVector{result}, ParameterVector{input_params});
}

static void get_transpose_data(const Shape& input_shape, const Shape& transpose_order,
                               ov::Output<ov::Node> input_node, TransposeData& transpose_data) {
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

    transpose_data.input = input_node;
    transpose_data.input_size = input_size;
    transpose_data.input_shape = input_shape;
    transpose_data.order = transpose_order;
}

static std::shared_ptr<Node> decompose_transpose_type1(const TransposeData& transpose_data) {
    size_t H_new = transpose_data.H;
    size_t W_new = transpose_data.W;

    if (transpose_data.order == Shape{0, 3, 1, 2} ||
        transpose_data.order == Shape{2, 0, 1}) {
        H_new = transpose_data.C * transpose_data.H;
    } else if (transpose_data.order == Shape{0, 2, 3, 1} ||
               transpose_data.order == Shape{1, 2, 0}) {
        H_new = transpose_data.C;
        W_new = transpose_data.H * transpose_data.W;
    } else if ((transpose_data.order == Shape{0, 1, 3, 2} ||
                transpose_data.order == Shape{0, 2, 1} ||
                transpose_data.order == Shape{1, 0}) &&
               transpose_data.C == 1) {
    } else
        return nullptr;


    Shape output_shape(transpose_data.input_size);
    auto i = 0;

    for (const auto& input_dim : transpose_data.order) {
        output_shape[i++] = transpose_data.input_shape[input_dim];
    }

    //const auto output_shape = transpose->output(0).get_shape();

    // GNA-compatible transpose
    if ((H_new != transpose_data.H || W_new != transpose_data.W) &&
        (H_new <= GNALimitations::transposeMaxMinDim || W_new <= GNALimitations::transposeMaxMinDim)) {
        auto reshape =
            std::make_shared<opset9::Reshape>(transpose_data.input,
                                              opset9::Constant::create(element::i32, Shape{2}, {H_new, W_new}),
                                              false);
        auto transpose_new =
            std::make_shared<op::Transpose>(reshape, opset9::Constant::create(element::i32, Shape{2}, {1, 0}));
        reshape = std::make_shared<opset9::Reshape>(
            transpose_new,
            opset9::Constant::create(element::i32, Shape{output_shape.size()}, output_shape)->output(0),
            false);
        return reshape;

        // GNA-incompatible transpose
    } else if ((H_new % GNALimitations::transposeMaxMinDim) == 0) {
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
            if (factors[i] > GNALimitations::transposeMaxMinDim) {
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
                auto reshape = std::make_shared<opset9::Reshape>(upstream[0], opset9::Constant::create(element::i32,
                    Shape{2}, {H_new * W_new / combined_factors[i], combined_factors[i]}), false);
                auto transpose = std::make_shared<op::Transpose>(reshape, opset9::Constant::create(element::i32, Shape{2}, {1, 0}));
                upstream[0] = transpose;
            }

            auto reshape = std::make_shared<opset9::Reshape>(upstream[0],
                opset9::Constant::create(element::i32, Shape{output_shape.size()}, output_shape), false);
            return reshape;
        }
    }
    return nullptr;
}

static std::shared_ptr<Node> decompose_transpose_type2(const TransposeData& transpose_data) {
    if (transpose_data.H <= 8 && transpose_data.W % GNALimitations::transposeMaxMinDim == 0 &&
        transpose_data.C * transpose_data.W <= GNALimitations::transposeMaxSize &&
        (transpose_data.order == Shape{0, 2, 1, 3} ||
         transpose_data.order == Shape{1, 0, 2})) {
        // We only allow decomposition of small problems to prevent creation of large number of layers
        if (transpose_data.C > 16)
            return nullptr;

        // Split into separate HxW matrices
        size_t axis = transpose_data.input_size - 3;
        const auto axis_node = opset9::Constant::create(element::i32, Shape{}, {axis});
        const auto split = std::make_shared<opset9::Split>(transpose_data.input, axis_node, transpose_data.C);

        OutputVector chunks;

        for (size_t c = 0; c < transpose_data.C; c++) {
            auto reshape = std::make_shared<opset9::Reshape>(split->output(c),
                opset9::Constant::create(element::i32, Shape{2}, {transpose_data.H, transpose_data.W}), false);
            auto transpose_part = std::make_shared<opset9::Transpose>(reshape, opset9::Constant::create(element::i32, Shape{2}, {1, 0}));
            chunks.push_back(transpose_part);
        }

        auto concat = std::make_shared<opset9::Concat>(chunks, 0);
        auto transpose_final =
            std::make_shared<opset9::Transpose>(concat, opset9::Constant::create(element::i32, Shape{2}, {1, 0}));
        auto reshape = (transpose_data.input_size == 3) ? std::make_shared<opset9::Reshape>(transpose_final,
            opset9::Constant::create(element::i32, Shape{3}, {transpose_data.H, transpose_data.C, transpose_data.W}), false)
            : std::make_shared<opset9::Reshape>(transpose_final, opset9::Constant::create(element::i32, Shape{4},
            {transpose_data.N, transpose_data.H, transpose_data.C, transpose_data.W}), false);
        return reshape;
    }

    return nullptr;
}

static std::shared_ptr<Node> decompose_transpose_type3(const TransposeData& transpose_data) {
    if (transpose_data.order == Shape{0, 1, 3, 2} ||
        transpose_data.order == Shape{0, 2, 1} || transpose_data.order == Shape{1, 0}) {
        OutputVector chunks;

        // Split to 2D transpose case
        if (transpose_data.C > 1) {
            size_t axis = transpose_data.input_size - 3;
            const auto axis_node = opset9::Constant::create(element::i32, Shape{}, {axis});
            auto split = std::make_shared<opset9::Split>(transpose_data.input, axis_node, transpose_data.C);

            for (size_t c = 0; c < transpose_data.C; c++) {
                chunks.push_back(split->output(c));
            }
        } else {
            chunks.push_back(transpose_data.input);
        }

        auto transpose_const = std::make_shared<opset9::Constant>(element::i32, Shape{sizeof(transpose_data.order)}, transpose_data.order);

        OutputVector transpose_parts;
        for (size_t c = 0; c < transpose_data.C; c++) {
            if (transpose_data.H <= 8) {
                //std::shared_ptr<Node> transpose_const = GetTransposeOrder(transpose_data);
                auto new_transpose = std::make_shared<op::Transpose>(chunks[c], transpose_const);
                transpose_parts.push_back(new_transpose->output(0));
            } else if (transpose_data.H % GNALimitations::transposeMaxMinDim == 0 &&
                       transpose_data.W % GNALimitations::transposeMaxMinDim == 0) {
                // We only allow decomposition of small problems to prevent creation of large number of layers
                auto const total_no_of_layers = (transpose_data.H * transpose_data.W) / 64 +
                                                2 * transpose_data.H / GNALimitations::transposeMaxMinDim;

                if (total_no_of_layers > 24)
                    return nullptr;

                // Split input into row blocks of height 8
                size_t H_div_8 = transpose_data.H / GNALimitations::transposeMaxMinDim;
                size_t W_div_8 = transpose_data.W / GNALimitations::transposeMaxMinDim;
                size_t axis = transpose_data.input_size - 2;
                const auto axis_node = opset9::Constant::create(element::i32, Shape{}, {axis});
                auto rowblock = std::make_shared<opset9::Split>(chunks[c], axis_node, H_div_8);

                // Transpose row blocks
                std::vector<OutputVector> subblock;

                for (size_t i = 0; i < H_div_8; i++) {
                    //std::shared_ptr<Node> transpose_const = GetTransposeOrder(transpose_data);
                    auto new_transpose = std::make_shared<op::Transpose>(rowblock->output(i), transpose_const);

                    // Split transposed row blocks into 8x8 blocks
                    OutputVector block;
                    auto split_rowblock = std::make_shared<opset9::Split>(new_transpose->output(0), axis_node, W_div_8);

                    for (size_t j = 0; j < W_div_8; j++) {
                        block.push_back(split_rowblock->output(j));
                    }

                    subblock.push_back(block);
                }

                // Un-transpose all 8x8 blocks
                for (size_t i = 0; i < H_div_8; i++) {
                    for (size_t j = 0; j < W_div_8; j++) {
                        //std::shared_ptr<Node> transpose_const = GetTransposeOrder(transpose_data);
                        auto new_transpose = std::make_shared<op::Transpose>(subblock[i][j], transpose_const);
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

                    auto concat = std::make_shared<opset9::Concat>(blocks, transpose_data.input_size - 2);
                    colblock.push_back(concat->output(0));
                }

                // Transpose each block column
                OutputVector transposed_rowblock;
                for (size_t j = 0; j < W_div_8; j++) {
                    //std::shared_ptr<Node> transpose_const = GetTransposeOrder(transpose_data);
                    auto new_transpose = std::make_shared<op::Transpose>(colblock[j], transpose_const);
                    transposed_rowblock.push_back(new_transpose->output(0));
                }

                // Contenate to form single final transposed tensor
                if (transposed_rowblock.size() > 1) {
                    auto concat = std::make_shared<opset9::Concat>(transposed_rowblock, transpose_data.input_size - 2);
                    transpose_parts.push_back(concat->output(0));
                } else {
                    transpose_parts.push_back(transposed_rowblock[0]);
                }
            }
        }

        std::shared_ptr<Node> last_node;

        // Contenate parts to form final transposed tensor
        if (transpose_data.C > 1) {
            size_t axis = transpose_data.input_size - 3;
            last_node = std::make_shared<opset9::Concat>(transpose_parts, axis);
        } else {
            last_node = transpose_parts[0].get_node_shared_ptr();
        }

        return last_node;
    }

    return nullptr;
}

std::shared_ptr<Function> create_reference_function(const Shape& input_shape,
                                                  const Shape& transpose_order) {
    auto input_params = std::make_shared<opset9::Parameter>(element::i32, input_shape);
    auto transpose_params = std::make_shared<opset9::Constant>(element::i32,
                                                                       Shape{transpose_order.size()},
                                                                       transpose_order);

    TransposeData transpose_data = {};
    std::shared_ptr<Node> transpose;

    get_transpose_data(input_shape, transpose_order, input_params, transpose_data);

    transpose = decompose_transpose_type1(transpose_data);

    if (transpose == nullptr) {
        transpose = decompose_transpose_type2(transpose_data);
    }

    if (transpose == nullptr) {
        transpose = decompose_transpose_type3(transpose_data);
    }

    auto result = std::make_shared<opset9::Result>(transpose);
    return std::make_shared<Function>(ResultVector{result}, ParameterVector{input_params});
}

} // namespace decomposeTranspose

// ---------------------------------------------------------------------------------------------------------------------

using FixtureInputShapes = std::pair<Shape /* input data */, Shape /* transpose order */>;

class DecomposeTransposeFixture
    : public CommonTestUtils::TestsCommon,
      public ::testing::WithParamInterface<FixtureInputShapes> {
public:
    void SetUp() override;

public:
    std::shared_ptr<Function> function, reference_function;
};

void DecomposeTransposeFixture::SetUp() {
    FixtureInputShapes input_shapes = this->GetParam();

    function = decomposeTranspose::create_function(input_shapes.first, input_shapes.second);
    reference_function = decomposeTranspose::create_reference_function(input_shapes.first, input_shapes.second);
}

void execute_test(std::shared_ptr<Function> function, std::shared_ptr<Function> reference_function) {
    pass::Manager manager, manager_ref;
    manager.register_pass<pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::Serialize>("ir0.xml", "ir0.bin", ngraph::pass::Serialize::Version::IR_V10);
    manager.register_pass<GNAPluginNS::DecomposeTranspose>();
    manager.register_pass<ngraph::pass::Serialize>("ir1.xml", "ir1.bin", ngraph::pass::Serialize::Version::IR_V10);
    manager.run_passes(function);
    manager_ref.register_pass<ngraph::pass::Serialize>("ir_ref.xml", "ir_ref.bin", ngraph::pass::Serialize::Version::IR_V10);
    manager_ref.run_passes(reference_function);
    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid);
}

std::vector<FixtureInputShapes> input_shapes = {{{1, 2, 4, 32}, {0, 3, 1, 2}},
                                                {{1, 1, 32, 64}, {0, 1, 3, 2}},
                                                {{1, 16, 8, 32}, {0, 2, 1, 3}},
                                                {{1, 2, 4, 32}, {0, 1, 3, 2}}};

TEST_P(DecomposeTransposeFixture, CompareFunctions) {
    execute_test(function, reference_function);
}

INSTANTIATE_TEST_SUITE_P(DecomposeTransposeTestSuite,
                         DecomposeTransposeFixture,
                         ::testing::ValuesIn(input_shapes));
