// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <queue>
#include <map>

#include "transformations/init_node_info.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

using namespace ngraph;
using namespace opset1;

namespace LayerTestsDefinitions {

enum class modelType {
    TranspConvTransp = 0,               /* Transpose(NHWC->NCHW) => conv => Transpose(NCHW->NHWC) */
    TranspConvBcastAddTransp,           /* Transpose(NHWC->NCHW) => conv => broadcasted add (BIAS) => Transpose(NCHW->NHWC) */
    TranspConvBcastAddMaxPoolTransp,    /* Transpose(NHWC->NCHW) => conv => broadcasted add (BIAS) => MaxPooling => Transpose(NCHW->NHWC) (2d max pool case) */
    TranspConvBcastAddActTransp,        /* Transpose(NHWC->NCHW) => conv => broadcasted add (BIAS) => ActivationFunction => Transpose(NCHW->NHWC) */
    TranspConvBcastAddMaxPoolActTransp, /* Transpose(NHWC->NCHW) => conv => broadcasted add (BIAS) => MaxPool => ActivationFunction => Transpose(NCHW->NHWC) */
    TranspConvTranspBcastAdd,           /* Transpose(NHWC->NCHW) => conv => Transpose(NCHW->NHWC) => BIAS (output of MO --disable_nhwc_to_nchw option) */
    TranspConvTranspBcastAddAct         /* Transpose(NHWC->NCHW) => conv => Transpose(NCHW->NHWC) => BIAS => AF (output of MO --disable_nhwc_to_nchw option) */
};

typedef std::tuple<
    InferenceEngine::SizeVector,    // Kernel size
    InferenceEngine::SizeVector,    // Strides
    std::vector<ptrdiff_t>,         // Pad begin
    std::vector<ptrdiff_t>,         // Pad end
    InferenceEngine::SizeVector,    // Dilation
    size_t,                         // Num out channels
    op::PadType                     // Padding type
> convSpecificParams;

typedef std::tuple<
    InferenceEngine::SizeVector,    // Bias
    InferenceEngine::SizeVector,    // Transposed Bias
    InferenceEngine::SizeVector,    // Maxpool pool
    InferenceEngine::SizeVector     // Maxpool strides
> miscSpecificParams;

typedef std::tuple<
    convSpecificParams,                 // Convolution parameters
    miscSpecificParams,                 // Bias & Maxpool parameters
    InferenceEngine::Precision,         // Network Precision
    std::string,                        // Target Device
    std::map<std::string, std::string>, // Configuration
    InferenceEngine::SizeVector,        // Input shapes
    modelType                           // Test model
> conv2DDecomposeParams;

class Conv2DDecomposeTest : public testing::WithParamInterface<conv2DDecomposeParams>,
    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<conv2DDecomposeParams> obj) {
        convSpecificParams convParams;
        miscSpecificParams miscParams;
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        InferenceEngine::SizeVector inputShape;
        modelType model;
        std::tie(convParams, miscParams, netPrecision, targetDevice, configuration, inputShape, model) = obj.param;
        op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation, bias, transpBias, maxpool_pool, maxpool_stride;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t numOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, numOutChannels, padType) = convParams;
        std::tie(bias, transpBias, maxpool_pool, maxpool_stride) = miscParams;

        std::ostringstream result;
        result << "M=" << static_cast<uint32_t>(model) << "_";
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "K" << CommonTestUtils::vec2str(kernel) << "_";
        result << "S" << CommonTestUtils::vec2str(stride) << "_";
        result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
        result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
        result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
        result << "O=" << numOutChannels << "_";
        result << "AP=" << padType << "_";
        result << "B=" << CommonTestUtils::vec2str(bias) << "_";
        result << "B=" << CommonTestUtils::vec2str(transpBias) << "_";
        result << "MPP=" << CommonTestUtils::vec2str(maxpool_pool) << "_";
        result << "MPS=" << CommonTestUtils::vec2str(maxpool_stride) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        return result.str();
    }

protected:
    void SetUp() override {
        threshold = 0.015;
        convSpecificParams convParams;
        miscSpecificParams miscParams;
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        modelType model;
        std::tie(convParams, miscParams, netPrecision, targetDevice, configuration, inputShape, model) = this->GetParam();
        op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation, bias, transpBias, maxpool_pool, maxpool_stride;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t numOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, numOutChannels, padType) = convParams;
        std::tie(bias, transpBias, maxpool_pool, maxpool_stride) = miscParams;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        Shape bias_shape{ bias };
        Shape transp_bias_shape{ transpBias };
        Shape maxpool_shape{ maxpool_pool };
        Strides maxpool_strides{ maxpool_stride };

        auto input = builder::makeParams(ngPrc, { inputShape });
        auto transpose_in_order = op::Constant::create(element::i64, Shape{ 4 }, { 0, 3, 1, 2 });
        auto transpose_in = std::make_shared<Transpose>(input[0], transpose_in_order);
        auto filter_size = std::accumulate(std::begin(kernel), std::end(kernel), 1, std::multiplies<size_t>());
        auto filter_weights = CommonTestUtils::generate_float_numbers(numOutChannels * inputShape[3] * filter_size, -0.03f, 0.03f);
        auto conv = builder::makeConvolution(transpose_in, ngPrc, kernel, stride, padBegin,
            padEnd, dilation, padType, numOutChannels, false, filter_weights);
        auto transpose_out_order = op::Constant::create(element::i64, Shape{ 4 }, { 0, 2, 3, 1 });
        auto bias_weights = CommonTestUtils::generate_float_numbers(shape_size(bias_shape), -1.5f, 1.5f);
        Output<Node> bias_const = std::make_shared<Constant>(ngPrc, bias_shape, bias_weights);
        Output<Node> transp_bias_const = std::make_shared<Constant>(ngPrc, transp_bias_shape, bias_weights);
        Output<Node> last_op = std::make_shared<Transpose>(conv, transpose_out_order);;

        switch (model) {
        case modelType::TranspConvBcastAddTransp:
        {
            auto bias = std::make_shared<Add>(conv, bias_const);
            last_op = std::make_shared<Transpose>(bias, transpose_out_order);
        }
        break;

        case modelType::TranspConvBcastAddMaxPoolTransp:
        {
            auto bcast_add = std::make_shared<Add>(conv, bias_const);
            auto maxpool = std::make_shared<MaxPool>(bcast_add, maxpool_strides, Shape{ 0, 0 }, Shape{ 0, 0 }, maxpool_shape,
                op::RoundingType::FLOOR, op::PadType::VALID);
            last_op = std::make_shared<Transpose>(maxpool, transpose_out_order);
        }
        break;

        case modelType::TranspConvBcastAddActTransp:
        {
            auto bcast_add = std::make_shared<Add>(conv, bias_const);
            auto activation = std::make_shared<Sigmoid>(bcast_add);
            last_op = std::make_shared<Transpose>(activation, transpose_out_order);
        }
        break;

        case modelType::TranspConvBcastAddMaxPoolActTransp:
        {
            auto bcast_add = std::make_shared<Add>(conv, bias_const);
            auto max_pool = std::make_shared<MaxPool>(bcast_add, Strides{ 1, 1 }, Shape{ 0, 0 }, Shape{ 0, 0 }, maxpool_shape,
                op::RoundingType::FLOOR, op::PadType::VALID);
            auto activation = std::make_shared<Relu>(max_pool);
            last_op = std::make_shared<Transpose>(activation, transpose_out_order);
        }
        break;

        case modelType::TranspConvTranspBcastAdd:
        {
            last_op = std::make_shared<Add>(last_op, transp_bias_const);
        }
        break;

        case modelType::TranspConvTranspBcastAddAct:
        {
            auto bcast_add = std::make_shared<Add>(last_op, transp_bias_const);
            last_op = std::make_shared<Sigmoid>(bcast_add);
        }
        break;

        case modelType::TranspConvTransp:
        default:
            break;
        }

        auto result = std::make_shared<Result>(last_op);
        function = std::make_shared<Function>(ResultVector{ result }, ParameterVector{ input });
    }
};

TEST_P(Conv2DDecomposeTest, CompareWithRefs) {
    Run();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    //TODO: FP16 is currently not supported by the transform
    //InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "1"},
        {"GNA_EXEC_TARGET", "GNA_TARGET_2_0"}
    }
};

const std::vector<op::PadType> padTypes = {
        op::PadType::EXPLICIT,
        op::PadType::SAME_LOWER,
        op::PadType::SAME_UPPER,
        op::PadType::VALID
};

const std::vector<modelType> models = {
    modelType::TranspConvTransp,
    modelType::TranspConvBcastAddTransp,
    modelType::TranspConvBcastAddActTransp,
    modelType::TranspConvTranspBcastAdd,
    modelType::TranspConvTranspBcastAddAct,
    //TODO: below scenarios are currently not supported
    //modelType::TranspConvBcastAddMaxPoolTransp,
    //modelType::TranspConvBcastAddMaxPoolActTransp,
};

const std::vector<std::vector<size_t>> input2DNHWC = { {1, 16, 16, 8} };
const std::vector<std::vector<size_t >> kernels2D = { {3, 2} };
const std::vector<std::vector<size_t >> strides2D = { {1, 1} };
const std::vector<std::vector<ptrdiff_t>> padBegins2D = { {1, 2} };
const std::vector<std::vector<ptrdiff_t>> padEnds2D = { {3, 1} };
//TODO: dilation != 1 fails due to accuracy
const std::vector<std::vector<size_t >> dilations2D = { {1, 1} };
const std::vector<size_t> numOutChannels2D = { 4 };
const std::vector<std::vector<size_t >> biases2D = { {1, 4, 1, 1} };
const std::vector<std::vector<size_t >> transp_biases2D = { {1, 1, 1, 4} };
//TODO: maxpool 2d is currently not supported by this transform
const std::vector<std::vector<size_t >> maxpool1D_pools = { {1, 2} };
const std::vector<std::vector<size_t >> maxpool1D_strides = { {1, 1} };

const auto conv2DParams = ::testing::Combine(
    ::testing::ValuesIn(kernels2D),
    ::testing::ValuesIn(strides2D),
    ::testing::ValuesIn(padBegins2D),
    ::testing::ValuesIn(padEnds2D),
    ::testing::ValuesIn(dilations2D),
    ::testing::ValuesIn(numOutChannels2D),
    ::testing::ValuesIn(padTypes)
);

const auto miscParams = ::testing::Combine(
    ::testing::ValuesIn(biases2D),
    ::testing::ValuesIn(transp_biases2D),
    ::testing::ValuesIn(maxpool1D_pools),
    ::testing::ValuesIn(maxpool1D_strides)
);

INSTANTIATE_TEST_CASE_P(smoke_2DConvDecompose, Conv2DDecomposeTest,
    ::testing::Combine(
        conv2DParams,
        miscParams,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs),
        ::testing::ValuesIn(input2DNHWC),
        ::testing::ValuesIn(models)),
    Conv2DDecomposeTest::getTestCaseName);

} // namespace LayerTestsDefinitions
