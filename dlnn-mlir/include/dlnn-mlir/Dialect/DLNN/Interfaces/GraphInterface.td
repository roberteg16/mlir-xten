//===- GraphInterface.td - DLNN graph interface ------------*- tablegen -*-===//
//
// This is the definitions file for the DLNN graph interface.
//
//===----------------------------------------------------------------------===//

#ifndef DLNN_GRAPHINTERFACE
#define DLNN_GRAPHINTERFACE

include "mlir/IR/OpBase.td"

def DLNN_Graph : OpInterface<"Graph"> {
    let cppNamespace = "::mlir::dlnn";

    let description = [{
        Interface for an operation that spans a DLNN dataflow graph.

        A graph is a collection of nodes and edges between them. See the Node
        interface for more details. Additionally, a graph may have inputs and
        outputs, which are edges within the graph that reach the outside.

        A graph operation must include a Block with "graph content" semantics
        (others may exist) and be `IsolatedFromAbove`. The contents block must
        have a terminator node.

        The content block arguments are the inputs to the graph, and the
        operands to the terminator node are the outputs to the graph.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/[{Gets the block with the graph contents.}],
            /*retTy=*/"::mlir::Block*",
            /*methodName=*/"getGraphContent",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                auto &region = $_op->getRegion(0);
                assert(region.hasOneBlock());
                return &region.front();
            }]
        >
    ];

    code verify = [{
        static_assert(
            ConcreteOp::template hasTrait<::mlir::OpTrait::IsIsolatedFromAbove>(),
            "`Graph` is only applicable to `IsolatedFromAbove` ops"
        );

        return graph_defaults::verify($_op);
    }];

    code extraSharedClassDeclaration = [{
        /// Gets the graph terminator node.
        ::mlir::dlnn::Node getGraphTerminator()
        {
            return cast<::mlir::dlnn::Node>(
                $_op.getGraphContent()->getTerminator());
        }

        /// Gets the graph input arguments.
        ::mlir::Block::BlockArgListType getGraphInputs()
        {
            return $_op.getGraphContent()->getArguments();
        }

        /// Gets the graph output operands.
        ::mlir::OperandRange getGraphOutputs()
        {
            return $_op.getGraphTerminator().getInputs();
        }
    }];
}

#endif
