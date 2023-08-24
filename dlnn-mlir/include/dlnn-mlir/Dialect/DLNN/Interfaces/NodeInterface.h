/// Declares the DLNN node interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#pragma once

#include "mlir/IR/Builders.h"

namespace mlir::dlnn {

//===----------------------------------------------------------------------===//
// Dataflow edges
//===----------------------------------------------------------------------===//

/// Represents a directed edge in a DLNN dataflow graph.
///
/// This is a thin handle around an operand or result of an operation.
///
/// An edge can be bound to a producer and a consumer. The default-initialized
/// edge is bound to neither. An edge that only has a consumer is an external
/// input, i.e. a BlockArgument. An edge that only has a producer is an external
/// output, i.e. a return operand.
class Edge {
    using operand_handle = OpOperand*;
    using result_handle = mlir::detail::OpResultImpl*;
    using handle_type = llvm::PointerUnion<operand_handle, result_handle>;

public:
    /// Detects an Edge for @p value .
    ///
    /// If @p value has exactly one use, the edge will be bound to the consumer.
    /// Otherwise, if @p value is the result of an operation, the edge will be
    /// bound to the producer. Otherwise, the edge is unbound.
    static Edge detect(Value value)
    {
        if (!value) return Edge();
        if (value.hasOneUse()) return {*value.getUses().begin().getOperand()};
        if (auto result = value.dyn_cast<OpResult>()) return Edge(result);
        return Edge();
    }

    /// Initializes an unbound edge.
    ///
    /// @post       `!isBound()`
    /*implicit*/ Edge() : m_handle(nullptr) {}
    /// Initializes an Edge from the operand at the consumer.
    ///
    /// If the operand is produced by another operation, the edge will be bound
    /// to both a consumer and a producer.
    ///
    /// @post       `isBound()`
    /*implicit*/ Edge(OpOperand &consumerOperand) : m_handle(&consumerOperand)
    {}
    /// Initializes an Edge from the result at the producer.
    ///
    /// The edge will not bind to any consumer, even if that is uniquely defined
    /// (i.e. the value has exactly one use). If that is your intent, use
    /// detect(Value) instead.
    ///
    /// @post       `isBound() == !!producerResult`
    explicit Edge(OpResult producerResult)
            : m_handle(ResultRange(producerResult).getBase())
    {}

    /// Determines whether this edge is bound to anything at all.
    bool isBound() const { return !m_handle.isNull(); }
    /// @copydoc isBound()
    operator bool() const { return isBound(); }

    /// Gets the value transferred along this edge.
    Value getValue() const
    {
        if (auto operand = getConsumerOperand()) return operand->get();
        return getProducerResult();
    }
    /// @copydoc getValue()
    operator Value() const { return getValue(); }

    /// Gets the OpResult at the producer, if bound.
    OpResult getProducerResult() const
    {
        if (auto result = m_handle.dyn_cast<result_handle>())
            return OpResult(result);
        return {};
    }
    /// Gets the producer, if bound.
    Operation* getProducer() const
    {
        if (auto result = getProducerResult()) return result.getOwner();
        return nullptr;
    }

    /// Gets the OpOperand at the consumer, if bound.
    OpOperand* getConsumerOperand() const
    {
        return m_handle.dyn_cast<operand_handle>();
    }
    /// Gets the consumer, if bound.
    Operation* getConsumer() const
    {
        if (auto operand = getConsumerOperand()) return operand->getOwner();
        return nullptr;
    }

private:
    handle_type m_handle;
};

/// Represents a range of dataflow edges on contiguous operands.
///
/// This is a wrapper around an OperandRange.
class EdgeRange : public llvm::detail::indexed_accessor_range_base<
                      EdgeRange,
                      OpOperand*,
                      Edge,
                      Edge,
                      Edge> {
public:
    using RangeBaseT::RangeBaseT;
    /*implicit*/ EdgeRange(OperandRange operands)
            : RangeBaseT(operands.getBase(), operands.size())
    {}

    /// Obtains the underlying OperandRange.
    OperandRange getOperands() const { return {getBase(), count}; }
    /// @copydoc getOperands()
    operator OperandRange() const { return getOperands(); }

    /// Obtains the underlying ValueRange.
    ValueRange getValues() const { return getOperands(); }
    /// @copydoc getValues()
    operator ValueRange() const { return getValues(); }

private:
    // See `llvm::detail::indexed_accessor_range_base` for details.
    static OpOperand* offset_base(OpOperand* object, ptrdiff_t index)
    {
        return object + index;
    }
    // See `llvm::detail::indexed_accessor_range_base` for details.
    static Edge dereference_iterator(OpOperand* object, ptrdiff_t index)
    {
        return {object[index]};
    }

    // Allow access to `offset_base` and `dereference_iterator`.
    friend RangeBaseT;
};

} // namespace mlir::dlnn

//===- Generated includes -------------------------------------------------===//

#include "dlnn-mlir/Dialect/DLNN/Interfaces/NodeInterface.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::dlnn {

//===----------------------------------------------------------------------===//
// Incoming edges & predecessors
//===----------------------------------------------------------------------===//

/// Iterator over incoming edges to a Node.
///
/// The iterator does not ensure that the edges are bound to a producer, which
/// happens when that lies outside of the graph.
struct IncomingEdgeIterator : llvm::indexed_accessor_iterator<
                                  IncomingEdgeIterator,
                                  Node,
                                  Edge,
                                  const Edge*,
                                  Edge> {
    /// Gets the first incoming edge iterator for @p node .
    static IncomingEdgeIterator begin(Node node)
    {
        if (!node) return {};
        return IncomingEdgeIterator(node, 0);
    }
    /// Gets the last incoming edge iterator for @p node .
    static IncomingEdgeIterator end(Node node)
    {
        if (!node) return {};
        return IncomingEdgeIterator(node, node.getNumInputs());
    }

    /*implicit*/ IncomingEdgeIterator() : indexed_accessor_iterator(Node{}, 0)
    {}
    /*implicit*/ IncomingEdgeIterator(const IncomingEdgeIterator &) = default;

    Edge operator*() const
    {
        return const_cast<Node &>(base).getInput(getIndex());
    }

private:
    using indexed_accessor_iterator::indexed_accessor_iterator;
};

/// Gets the range of incoming edges of @p node .
///
/// See IncomingEdgeIterator for more information.
inline llvm::iterator_range<IncomingEdgeIterator> incomingEdges(Node node)
{
    return {IncomingEdgeIterator::begin(node), IncomingEdgeIterator::end(node)};
}

/// Iterator over predecessors of a Node.
///
/// The iterator does not ensure that the returned predecessor nodes are unique,
/// nor that they are non-null. The latter happens when a predecessor is opaque
/// or lies outside the graph.
class PredecessorIterator : public llvm::iterator_adaptor_base<
                                PredecessorIterator,
                                IncomingEdgeIterator,
                                std::forward_iterator_tag,
                                Node> {
    struct PointerProxy {
        mutable Node node;
        Node* operator->() const { return &node; }
    };

public:
    /// Gets the first predecessor iterator for @p node .
    static PredecessorIterator begin(Node node)
    {
        return PredecessorIterator(IncomingEdgeIterator::begin(node));
    }
    /// Gets the last predecessor iterator for @p node .
    static PredecessorIterator end(Node node)
    {
        return PredecessorIterator(IncomingEdgeIterator::end(node));
    }

    using iterator_adaptor_base::iterator_adaptor_base;

    Node operator*() const
    {
        return dyn_cast_or_null<Node>(wrapped()->getProducer());
    }
    PointerProxy operator->() const { return {**this}; }
};

/// Gets the range of predecessors of @p node .
///
/// See PredecessorIterator for more information.
inline llvm::iterator_range<PredecessorIterator> predecessors(Node node)
{
    return {PredecessorIterator::begin(node), PredecessorIterator::end(node)};
}

//===----------------------------------------------------------------------===//
// Outgoing edges & successors
//===----------------------------------------------------------------------===//

/// Iterator over incoming edges from a Node.
///
/// The iterator does not ensure that the edges are bound to a consumer, which
/// happens when that lies outside of the graph.
struct OutgoingEdgeIterator : llvm::iterator_adaptor_base<
                                  OutgoingEdgeIterator,
                                  ResultRange::UseIterator,
                                  std::forward_iterator_tag,
                                  Edge,
                                  std::ptrdiff_t,
                                  const Edge*,
                                  Edge> {
    /// Gets the first outgoing edge iterator for @p node .
    static OutgoingEdgeIterator begin(Node node)
    {
        if (!node) return {};
        return OutgoingEdgeIterator(node.getOutputs().getUses().begin());
    }
    /// Gets the last outgoing edge iterator for @p node .
    static OutgoingEdgeIterator end(Node node)
    {
        if (!node) return {};
        return OutgoingEdgeIterator(node.getOutputs().getUses().end());
    }

    /*implicit*/ OutgoingEdgeIterator()
            : iterator_adaptor_base(ResultRange::UseIterator(
                ResultRange(
                    static_cast<mlir::detail::OpResultImpl*>(nullptr),
                    0),
                true))
    {}
    /*implicit*/ OutgoingEdgeIterator(const OutgoingEdgeIterator &) = default;

    Edge operator*() const { return Edge(*wrapped()); }

private:
    using iterator_adaptor_base::iterator_adaptor_base;
};

/// Gets the range of outgoing edges of @p node .
///
/// See OutgoingEdgeIterator for more information.
inline llvm::iterator_range<OutgoingEdgeIterator> outgoingEdges(Node node)
{
    return {OutgoingEdgeIterator::begin(node), OutgoingEdgeIterator::end(node)};
}

/// Iterator over successors of a Node.
///
/// The iterator does not ensure that the returned successor nodes are unique,
/// nor that they are non-null. The latter happens when a successor is opaque
/// or lies outside the graph.
class SuccessorIterator : public llvm::iterator_adaptor_base<
                              SuccessorIterator,
                              OutgoingEdgeIterator,
                              std::forward_iterator_tag,
                              Node> {
    struct PointerProxy {
        mutable Node node;
        Node* operator->() const { return &node; }
    };

public:
    /// Gets the first successor iterator for @p node .
    static SuccessorIterator begin(Node node)
    {
        return SuccessorIterator(OutgoingEdgeIterator::begin(node));
    }
    /// Gets the last successor iterator for @p node .
    static SuccessorIterator end(Node node)
    {
        return SuccessorIterator(OutgoingEdgeIterator::end(node));
    }

    using iterator_adaptor_base::iterator_adaptor_base;

    Node operator*() const
    {
        return dyn_cast_or_null<Node>(wrapped()->getProducer());
    }
    PointerProxy operator->() const { return {**this}; }
};

/// Gets the range of successors of @p node .
///
/// See SuccessorIterator for more information.
inline llvm::iterator_range<SuccessorIterator> successors(Node node)
{
    return {SuccessorIterator::begin(node), SuccessorIterator::end(node)};
}

} // namespace mlir::dlnn
