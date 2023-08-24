/// Implements the DLNN dialect graph interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#include "dlnn-mlir/Dialect/DLNN/Interfaces/GraphInterface.h"

#include "dlnn-mlir/Dialect/DLNN/IR/DLNN.h"
#include "dlnn-mlir/Dialect/DLNN/Utils/STLExtras.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::dlnn;

LogicalResult mlir::dlnn::graph_defaults::verify(Operation* op)
{
    auto self = cast<Graph>(op);

    // Accept empty graphs.
    if (!self.getGraphContent()) return success();

    // Graph must be terminated by a node.
    if (self.getGraphContent()->empty())
        return op->emitOpError() << "graph has no terminator";
    auto terminator = &self.getGraphContent()->back();
    if (!isa<Node>(terminator))
        return terminator->emitOpError() << "graph terminator must be a Node";

    return success();
}

LogicalResult mlir::dlnn::graph_defaults::verifyUse(
    function_ref<InFlightDiagnostic()> emitError,
    Graph graph,
    TypeRange inputs,
    TypeRange outputs)
{
    assert(graph);

    // The inputs must match.
    const auto graphIns = TypeRange(graph.getGraphInputs());
    if (inputs.size() != graphIns.size())
        return emitError() << "number of inputs (" << inputs.size()
                           << ") does not match number of graph inputs ("
                           << graphIns.size() << ")";
    for (auto [idx, in] : enumerate(graphIns))
        if (inputs[idx] != in)
            return emitError()
                   << "type of input #" << idx << " (" << inputs[idx]
                   << ") does not match graph input type (" << in << ")";

    // The outputs must match.
    const auto graphOuts = graph.getGraphOutputs().getTypes();
    if (outputs.size() != graphOuts.size())
        return emitError() << "number of outputs (" << outputs.size()
                           << ") does not match number of graph outputs ("
                           << graphOuts.size() << ")";
    for (auto [idx, out] : enumerate(graphOuts))
        if (outputs[idx] != out)
            return emitError()
                   << "type of output #" << idx << " (" << outputs[idx]
                   << ") does not match graph output type (" << out << ")";

    return success();
}

void NodesIterator::scanToNextNode()
{
    while (!m_stack.empty()) {
        // Return from subgraphs.
        if (m_stack.back().empty()) {
            m_stack.pop_back();
            continue;
        }

        auto next = *m_stack.back();

        // Recurse into subgraphs.
        if (auto subgraph = dyn_cast<Graph>(next)) {
            // We will return to the next op after the graph op.
            ++m_stack.back();
            // Enter the subgraph and continue scanning from there.
            if (auto content = subgraph.getGraphContent())
                m_stack.emplace_back(*content);
            continue;
        }

        // Terminate on Node
        if (auto node = dyn_cast<Node>(next)) return;

        // Continue scanning.
        ++m_stack.back();
    }

    // Reached the end of the graph.
}

//===- Generated implementation -------------------------------------------===//

#include "dlnn-mlir/Dialect/DLNN/Interfaces/GraphInterface.cpp.inc"

//===----------------------------------------------------------------------===//
