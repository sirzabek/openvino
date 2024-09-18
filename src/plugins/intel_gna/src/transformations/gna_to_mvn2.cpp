/* ============================================================================
 * INTEL CONFIDENTIAL
 *
 * Copyright 2021 Intel Corporation All Rights Reserved.
 *
 * The source code contained or described herein and all documents related to
 * the source code ("Material") are owned by Intel Corporation or its suppliers
 * or licensors. Title to the Material remains with Intel Corporation or its
 * suppliers and licensors. The Material contains trade secrets and proprietary
 * and confidential information of Intel or its suppliers and licensors. The
 * Material is protected by worldwide copyright and trade secret laws and
 * treaty provisions. No part of the Material may be used, copied, reproduced,
 * modified, published, uploaded, posted, transmitted, distributed, or
 * disclosed in any way without Intel's prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other intellectual
 * property right is granted to or conferred upon you by disclosure or delivery
 * of the Materials, either expressly, by implication, inducement, estoppel or
 * otherwise. Any license under such intellectual property rights must be
 * express and approved by Intel in writing.
 * ============================================================================
 */

#include "gna_to_mvn2.hpp"
#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

using namespace ngraph;
using namespace op;

// Converts custom normalization pattern into MVN layer
// Pattern:  Reshape --------------> Subtract ----------------------------------------------->
//                   --> ReduceMean --/                                                         \
//                   --------------> Subtract --> Multiply --> ReduceMean --> Add --> Sqrt --> Divide -->
//                   --> ReduceMean --/

bool ngraph::pass::GnaCustomToMvn::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto reshape = std::dynamic_pointer_cast<ngraph::opset1::Reshape>(node);
        auto transpose = std::dynamic_pointer_cast<ngraph::opset1::Transpose>(node);
        if ((nullptr == reshape) && (nullptr == transpose)) {
            continue;
        }

        // Ugly code to match our pattern
        std::shared_ptr<ngraph::opset1::Subtract> subtract1 = nullptr;
        std::shared_ptr<ngraph::opset1::Subtract> subtract2 = nullptr;
        std::shared_ptr<ov::op::v0::FakeQuantize> subtractfq1 = nullptr;
        std::shared_ptr<ov::op::v0::FakeQuantize> subtractfq2 = nullptr;
        std::shared_ptr<ngraph::opset1::ReduceMean> reducemean1 = nullptr;
        std::shared_ptr<ngraph::opset1::ReduceMean> reducemean2 = nullptr;
        std::shared_ptr<ngraph::opset1::ReduceMean> reducemean3 = nullptr;
        std::shared_ptr<ov::op::v0::FakeQuantize> reducemeanfq1 = nullptr;
        std::shared_ptr<ov::op::v0::FakeQuantize> reducemeanfq2 = nullptr;
        std::shared_ptr<ov::op::v0::FakeQuantize> reducemeanfq3 = nullptr;
        std::shared_ptr<ov::op::v1::Multiply> multiply1 = nullptr;
        std::shared_ptr<ov::op::v0::FakeQuantize> multiplyfq1 = nullptr;
        std::shared_ptr<ngraph::opset1::Add> add1 = nullptr;
        std::shared_ptr<ov::op::v0::FakeQuantize> addfq1 = nullptr;
        std::shared_ptr<ngraph::opset1::Sqrt> sqrt1 = nullptr;
        std::shared_ptr<ov::op::v1::Divide> divide1 = nullptr;
        std::shared_ptr<ov::op::v0::FakeQuantize> dividefq1 = nullptr;
        std::set<ov::Input<ov::Node>> children;
        if (reshape) {
            children = reshape->output(0).get_target_inputs();
        } else {
            children = transpose->output(0).get_target_inputs();
        }
        if (children.size() != 4) {
            continue;
        }
        std::vector<std::shared_ptr<ngraph::opset1::Subtract>> subtract;
        for (auto p = children.begin(); p != children.end(); p++) {
            auto child_node = p->get_node()->shared_from_this();
            auto s_ptr = std::dynamic_pointer_cast<ngraph::opset1::Subtract>(child_node);
            if (s_ptr) {
                subtract.push_back(s_ptr);
            }
        }
        if (subtract.size() != 2) {
            continue;
        }
        auto children1 = subtract[0]->output(0).get_target_inputs();
        auto children2 = subtract[1]->output(0).get_target_inputs();
        subtract1 = subtract[0];
        subtract2 = subtract[1];
        subtractfq1 = std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(children1.begin()->get_node()->shared_from_this());
        subtractfq2 = std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(children2.begin()->get_node()->shared_from_this());
        if (subtractfq1) {
            auto fqchildren = subtractfq1->output(0).get_target_inputs();
            multiply1 = std::dynamic_pointer_cast<ov::op::v1::Multiply>(fqchildren.begin()->get_node()->shared_from_this());
        } else {
            multiply1 = std::dynamic_pointer_cast<ov::op::v1::Multiply>(children1.begin()->get_node()->shared_from_this());
        }
        if (subtractfq2) {
            auto fqchildren = subtractfq2->output(0).get_target_inputs();
            divide1 = std::dynamic_pointer_cast<ov::op::v1::Divide>(fqchildren.begin()->get_node()->shared_from_this());
        } else {
            divide1 = std::dynamic_pointer_cast<ov::op::v1::Divide>(children2.begin()->get_node()->shared_from_this());
        }
        if ((multiply1 == nullptr) && (divide1 == nullptr)) {
            subtract1 = subtract[1];
            subtract2 = subtract[0];
            children1 = subtract1->output(0).get_target_inputs();
            children2 = subtract2->output(0).get_target_inputs();
            subtractfq1 = std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(children1.begin()->get_node()->shared_from_this());
            subtractfq2 = std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(children2.begin()->get_node()->shared_from_this());
            if (subtractfq1) {
                auto fqchildren = subtractfq1->output(0).get_target_inputs();
                multiply1 = std::dynamic_pointer_cast<ov::op::v1::Multiply>(fqchildren.begin()->get_node()->shared_from_this());
            } else {
                multiply1 = std::dynamic_pointer_cast<ov::op::v1::Multiply>(children1.begin()->get_node()->shared_from_this());
            }
            if (subtractfq2) {
                auto fqchildren = subtractfq2->output(0).get_target_inputs();
                divide1 = std::dynamic_pointer_cast<ov::op::v1::Divide>(fqchildren.begin()->get_node()->shared_from_this());         
            } else {
                divide1 = std::dynamic_pointer_cast<ov::op::v1::Divide>(children2.begin()->get_node()->shared_from_this());
            }
        }
        if ((multiply1 == nullptr) || (divide1 == nullptr)) {
            continue;
        }
        reducemeanfq1 = std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(subtract1->input_value(1).get_node_shared_ptr());
        reducemeanfq2 = std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(subtract2->input_value(1).get_node_shared_ptr());
        reducemean1 = std::dynamic_pointer_cast<ngraph::opset1::ReduceMean>(subtract1->input_value(1).get_node_shared_ptr());
        reducemean2 = std::dynamic_pointer_cast<ngraph::opset1::ReduceMean>(subtract2->input_value(1).get_node_shared_ptr());
        if (reducemeanfq1) {
            reducemean1 = std::dynamic_pointer_cast<ngraph::opset1::ReduceMean>(reducemeanfq1->input_value(0).get_node_shared_ptr());
        }
        if (reducemeanfq2) {
            reducemean2 = std::dynamic_pointer_cast<ngraph::opset1::ReduceMean>(reducemeanfq2->input_value(0).get_node_shared_ptr());
        }
        if ((reducemean1 == nullptr) || (reducemean2 == nullptr)) {
            continue;
        }
        children = multiply1->output(0).get_target_inputs();
        multiplyfq1 = std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(children.begin()->get_node()->shared_from_this());
        reducemean3 = std::dynamic_pointer_cast<ngraph::opset1::ReduceMean>(children.begin()->get_node()->shared_from_this());
        if (multiplyfq1) {
            auto fqchildren = multiplyfq1->output(0).get_target_inputs();
            reducemean3 = std::dynamic_pointer_cast<ngraph::opset1::ReduceMean>(fqchildren.begin()->get_node()->shared_from_this());
        }
        if ((children.size() != 1) || (reducemean3 == nullptr)) {
            continue;
        }
        children = reducemean3->output(0).get_target_inputs();
        reducemeanfq3 = std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(children.begin()->get_node()->shared_from_this());
        add1 = std::dynamic_pointer_cast<ngraph::opset1::Add>(children.begin()->get_node()->shared_from_this());
        if (reducemeanfq3) {
            auto fqchildren = reducemeanfq3->output(0).get_target_inputs();
            add1 = std::dynamic_pointer_cast<ngraph::opset1::Add>(fqchildren.begin()->get_node()->shared_from_this());
        }
        if ((children.size() != 1) || (add1 == nullptr)) {
            continue;
        }
        children = add1->output(0).get_target_inputs();
        addfq1 = std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(children.begin()->get_node()->shared_from_this());
        sqrt1 = std::dynamic_pointer_cast<ngraph::opset1::Sqrt>(children.begin()->get_node()->shared_from_this());
        if (addfq1) {
            auto fqchildren = addfq1->output(0).get_target_inputs();
            sqrt1 = std::dynamic_pointer_cast<ngraph::opset1::Sqrt>(fqchildren.begin()->get_node()->shared_from_this());
        }
        if ((children.size() != 1) || (sqrt1 == nullptr)) {
            continue;
        }
        children1 = subtract2->output(0).get_target_inputs();
        children2 = sqrt1->output(0).get_target_inputs();
        auto tmp1 = std::dynamic_pointer_cast<ngraph::opset1::Divide>(children1.begin()->get_node()->shared_from_this());
        auto tmp2 = std::dynamic_pointer_cast<ngraph::opset1::Divide>(children2.begin()->get_node()->shared_from_this());
        if ((divide1 != tmp1) || (divide1 != tmp2)) {
            children1 = subtractfq2->output(0).get_target_inputs();
            tmp1 = std::dynamic_pointer_cast<ngraph::opset1::Divide>(children1.begin()->get_node()->shared_from_this());
            if ((divide1 != tmp1) || (divide1 != tmp2)) {
                continue;
            }
        }
        auto epsilon_constfq = std::dynamic_pointer_cast<ov::op::v0::FakeQuantize>(add1->input_value(1).get_node_shared_ptr());
        auto epsilon_const = std::dynamic_pointer_cast<ngraph::opset1::Constant>(add1->input_value(1).get_node_shared_ptr());
        if (epsilon_constfq) {
            epsilon_const = std::dynamic_pointer_cast<ngraph::opset1::Constant>(epsilon_constfq->input_value(0).get_node_shared_ptr());
        }
        if (epsilon_const == nullptr) {
            continue;
        }
        const float* epsilon_ptr = epsilon_const->get_data_ptr<float>();

        auto across_channels = false;
        auto normalize_variance = true;
        float epsilon = *epsilon_ptr;
        op::MVNEpsMode eps_mode = op::MVNEpsMode::INSIDE_SQRT;
        std::shared_ptr<ngraph::op::v6::MVN> mvn_1 = nullptr;
        if (reshape) {
            mvn_1 = std::make_shared<op::v6::MVN>(reshape->output(0),
                op::Constant::create(ngraph::element::i64, Shape{1}, {1})->output(0), normalize_variance, epsilon, eps_mode);
            ngraph::replace_node_update_name(divide1, mvn_1);
        } else {
            bool absorb_transposes = false;
            const Output<Node>& parent = transpose->input_value(0);
            auto input_shape = parent.get_shape();
            const Output<Node>& transpose_order = transpose->input_value(1);
            auto const_with_order_values = std::dynamic_pointer_cast<ngraph::opset1::Constant>(transpose_order.get_node_shared_ptr());
            if (const_with_order_values) {
                const int64_t* order = const_with_order_values->get_data_ptr<int64_t>();
                ov::Output<ov::Node> axes = reducemean1->input_value(1);
                auto axis_const_data = std::dynamic_pointer_cast<Constant>(axes.get_node_shared_ptr())->get_data_ptr();
                int64_t axis_value = *((int64_t*)axis_const_data);
                // if MVN axis is N-2 and transpose is simple 2D H<-->W then we can remove the transposes
                if ((axis_value == input_shape.size() - 2) && (order[input_shape.size()-1] == input_shape.size()-2) && (order[input_shape.size() - 2] == input_shape.size() - 1)) {
                    absorb_transposes = true;
                }
            }
            std::shared_ptr<ngraph::opset1::Transpose> transpose_after = nullptr;
            auto children_div = divide1->output(0).get_target_inputs();
            if (children_div.size() == 1) {
                transpose_after = std::dynamic_pointer_cast<ngraph::opset1::Transpose>(children_div.begin()->get_node()->shared_from_this());
                if (transpose_after) {
                    const Output<Node>& transpose_order = transpose_after->input_value(1);
                    auto const_with_order_values = std::dynamic_pointer_cast<ngraph::opset1::Constant>(transpose_order.get_node_shared_ptr());
                    if (const_with_order_values) {
                        const int64_t* order = const_with_order_values->get_data_ptr<int64_t>();
                        // if trailing transpose is not simple 2D H<-->W then we can't remove the transposes
                        if ((order[input_shape.size()-1] != input_shape.size()-2) || (order[input_shape.size() - 2] != input_shape.size() - 1)) {
                            absorb_transposes = false;
                        }
                    }
                } else {
                    absorb_transposes = false;
                }
            }
            if (absorb_transposes) {
                mvn_1 = std::make_shared<op::v6::MVN>(parent,
                    op::Constant::create(ngraph::element::i64, Shape{1}, {input_shape.size()-1})->output(0), normalize_variance, epsilon, eps_mode);
                ngraph::replace_node_update_name(transpose_after, mvn_1);
            } else {
                mvn_1 = std::make_shared<op::v6::MVN>(transpose->output(0),
                    op::Constant::create(ngraph::element::i64, Shape{1}, {1})->output(0), normalize_variance, epsilon, eps_mode);
                ngraph::replace_node_update_name(divide1, mvn_1);
            }
        }

        is_graph_modfied = true;
    }
    return is_graph_modfied;
}
