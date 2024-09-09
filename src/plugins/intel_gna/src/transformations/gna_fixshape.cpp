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

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

#include "gna_fixshape.hpp"

using namespace ngraph;
using namespace op;

// This works around an apparent OpenVINO bug.  Even when mo is run with the --static_shape
// parameter, it is possible that some parameters and reshapes will still have dynamic shapes.
// This causes the GNA graph compiler to fail in convert_function_to_cnn_network.  The 
// workaround is simply to replace all dynamic shapes with their static equivalent.

bool ngraph::pass::GnaShapeFixup::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto parameter = std::dynamic_pointer_cast<ov::op::v0::Parameter>(node);
        auto readvalue = std::dynamic_pointer_cast<ov::op::v6::ReadValue>(node);
        auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(node);
        if (parameter != nullptr) {
            auto output_shape = parameter->output(0).get_shape();
            parameter->set_partial_shape(output_shape);
            is_graph_modfied = true;
            continue;
        } else if (readvalue != nullptr) {
            auto variable = readvalue->get_variable();
            auto output_type = readvalue->output(0).get_element_type();
            auto output_shape = readvalue->output(0).get_shape();
            variable->update_data_shape(output_shape);
            variable->update_data_type(output_type);
            is_graph_modfied = true;
            continue;
        } else if (reshape != nullptr) {
            const Output<Node>& parent = reshape->input_value(0);
            auto input_shape = reshape->input_value(0).get_shape();
            auto output_shape = reshape->output(0).get_shape();
            auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(parent,
                op::Constant::create(ngraph::element::i64, Shape{output_shape.size()}, output_shape)->output(0), false);
            ngraph::replace_node_update_name(reshape, new_reshape);
            is_graph_modfied = true;
            continue;                    
        } else {
            continue;
        }

    }
    return is_graph_modfied;
}
