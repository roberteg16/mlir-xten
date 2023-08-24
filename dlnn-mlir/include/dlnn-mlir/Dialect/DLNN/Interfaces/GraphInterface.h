/// Declares the DLNN graph interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#pragma once

#include "dlnn-mlir/Dialect/DLNN/Enums.h"
#include "dlnn-mlir/Dialect/DLNN/Interfaces/NodeInterface.h"
#include "mlir/IR/Builders.h"

#include "llvm/ADT/GraphTraits.h"

namespace mlir::dlnn::graph_defaults {

/// Verifies a Graph op.
///
/// The verifier checks that the terminator of the graph content block is a
/// Node.
///
/// @pre        `isa<Graph>(op)`
LogicalResult verify(Operation* op);

} // namespace mlir::dlnn::graph_defaults

//===- Generated includes -------------------------------------------------===//

#include "dlnn-mlir/Dialect/DLNN/Interfaces/GraphInterface.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::dlnn {

namespace graph_defaults {

/// Verifies a usage of a Graph op.
///
/// The verifier checks that the intended @p inputs and @p outputs to @p graph
/// match its signature.
///
/// @pre        `graph`
LogicalResult verifyUse(
    function_ref<InFlightDiagnostic()> emitError,
    Graph graph,
    TypeRange inputs,
    TypeRange outputs);

} // namespace graph_defaults

/// Iterator over all nodes in a (nested) Graph.
///
/// This iterator will recurse into all embedded subgraphs and return their
/// nodes. Crucially, however, operations that are both Node and Graph will not
/// be yielded by this operator.
class NodesIterator : public llvm::iterator_facade_base<
                          NodesIterator,
                          std::forward_iterator_tag,
                          Node,
                          std::ptrdiff_t,
                          Node*,
                          Node> {
    struct PointerProxy {
        mutable Node node;
        Node* operator->() const { return &node; }
    };

    struct OpRange {
        /*implicit*/ OpRange(Block &block)
                : begin(block.begin()),
                  end(block.end())
        {}

        bool empty() const { return begin == end; }

        bool operator==(const OpRange &rhs) const { return begin == rhs.begin; }

        OpRange &operator++()
        {
            ++begin;
            return *this;
        }
        Operation* operator*() const { return &*begin; }

    private:
        Block::OpListType::iterator begin;
        Block::OpListType::iterator end;
    };

public:
    static NodesIterator begin(Graph graph) { return NodesIterator(graph); }
    static NodesIterator end(Graph) { return {}; }

    /// Initializes a NodesIterator sentinel.
    /*implicit*/ NodesIterator() = default;
    /// Initializes a NodesIterator for @p graph and its subgraphs.
    /*implicit*/ NodesIterator(Graph graph) : m_stack()
    {
        if (graph && graph.getGraphContent()) {
            // Scan to the first node in the graph.
            m_stack.emplace_back(*graph.getGraphContent());
            scanToNextNode();
        }
    }

    bool operator==(const NodesIterator &rhs) const
    {
        // This will also be true for all sentinels, i.e. iterators with empty
        // stacks.
        return m_stack == rhs.m_stack;
    }

    NodesIterator &operator++()
    {
        ++m_stack.back();
        scanToNextNode();
        return *this;
    }

    Node operator*() const
    {
        assert(!m_stack.empty() && "iterator not dereferencable");
        return cast<Node>(*m_stack.back());
    }
    PointerProxy operator->() const { return {**this}; }

private:
    void scanToNextNode();

    SmallVector<OpRange> m_stack;
};

/// Gets the range of all nodes in a (nested) Graph.
///
/// See NodesIterator for more information.
inline llvm::iterator_range<NodesIterator> nodes(Graph graph)
{
    return {NodesIterator::begin(graph), NodesIterator::end(graph)};
}

} // namespace mlir::dlnn

/// Implement GraphTraits for DLNN operator graphs in producer->consumer order.
///
/// See llvm::GraphTraits for more information.
template<>
struct llvm::GraphTraits<mlir::dlnn::Graph> {
    using NodeRef = mlir::dlnn::Node;

    // NOTE: We can't implement this unless we create a virtual "entry node" of
    //       all graph inputs, which is technically possible, but means that
    //       NodeRef has to become a PointerUnion _everywhere_.
    // static NodeRef getEntryNode(const GraphType &)

    using nodes_iterator = ::mlir::dlnn::NodesIterator;
    static nodes_iterator nodes_begin(mlir::dlnn::Graph* graph)
    {
        return nodes_iterator::begin(*graph);
    }
    static nodes_iterator nodes_end(mlir::dlnn::Graph* graph)
    {
        return nodes_iterator::end(*graph);
    }

    using ChildIteratorType = ::mlir::dlnn::SuccessorIterator;
    static ChildIteratorType child_begin(NodeRef node)
    {
        return ChildIteratorType::begin(node);
    }
    static ChildIteratorType child_end(NodeRef node)
    {
        return ChildIteratorType::end(node);
    }

    using EdgeRef = ::mlir::dlnn::Edge;
    using ChildEdgeIteratorType = ::mlir::dlnn::OutgoingEdgeIterator;
    static ChildEdgeIteratorType child_edge_begin(NodeRef node)
    {
        return ChildEdgeIteratorType::begin(node);
    }
    static ChildEdgeIteratorType child_edge_end(NodeRef node)
    {
        return ChildEdgeIteratorType::end(node);
    }
    static NodeRef edge_dest(EdgeRef edge)
    {
        return mlir::dyn_cast_or_null<NodeRef>(edge.getConsumer());
    }
};

/// Implement GraphTraits for DLNN operator graphs in consumer->producer order.
///
/// See llvm::GraphTraits for more information.
template<>
struct llvm::GraphTraits<llvm::Inverse<mlir::dlnn::Graph>> {
    using NodeRef = mlir::dlnn::Node;

    // NOTE: Since we required the graph terminator to be a node, there is an
    //       unambiguous entry point to every graph!
    static NodeRef getEntryNode(const mlir::dlnn::Graph &graph)
    {
        if (auto content =
                const_cast<mlir::dlnn::Graph &>(graph).getGraphContent())
            return cast<NodeRef>(content->getTerminator());

        return {};
    }

    using nodes_iterator = ::mlir::dlnn::NodesIterator;
    static nodes_iterator nodes_begin(mlir::dlnn::Graph* graph)
    {
        return nodes_iterator::begin(*graph);
    }
    static nodes_iterator nodes_end(mlir::dlnn::Graph* graph)
    {
        return nodes_iterator::end(*graph);
    }

    using ChildIteratorType = ::mlir::dlnn::PredecessorIterator;
    static ChildIteratorType child_begin(NodeRef node)
    {
        return ChildIteratorType::begin(node);
    }
    static ChildIteratorType child_end(NodeRef node)
    {
        return ChildIteratorType::end(node);
    }

    using EdgeRef = ::mlir::dlnn::Edge;
    using ChildEdgeIteratorType = ::mlir::dlnn::IncomingEdgeIterator;
    static ChildEdgeIteratorType child_edge_begin(NodeRef node)
    {
        return ChildEdgeIteratorType::begin(node);
    }
    static ChildEdgeIteratorType child_edge_end(NodeRef node)
    {
        return ChildEdgeIteratorType::end(node);
    }
    static NodeRef edge_dest(EdgeRef edge)
    {
        return mlir::dyn_cast_or_null<NodeRef>(edge.getProducer());
    }
};
