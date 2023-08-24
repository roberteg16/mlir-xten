//===- Types.td - DLNN type declarations -------------------*- tablegen -*-===//
//
// This is the definitions file for the DLNN dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef DLNN_TYPES
#define DLNN_TYPES

include "dlnn-mlir/Dialect/DLNN/IR/Base.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def DLNN_FeatureMapType : DLNN_Type<"FeatureMap"> {
    let mnemonic = "fm";
    let summary = "Feature map type";
    let description = [{
        A feature map is an organized type that arranges scalar values in a
        multi-dimensional shape, which has:

            - Exactly one static dimension with "channel" semantics `c`.
            - An arbitrary amount of static or dynamic dimensions with "size"
              semantics `x_i`.

        The canonical ordering of these dimensions is `c, x_0, ..., x_R`, where
        `R` is the number of dimensions with "size" semantics.

        Example:
        ```mlir
        // {numChannels: 1, scalarType: f16, size: [], org: id}
        !fm<f16>
        // {numChannels: 3, scalarType: f16, size: [108, 108], org: id}
        !fm<3xf16[108, 108]>
        // {numChannels: 3, scalarType: f16, size: [dyn, dyn], org: nchw}
        !fm<3xf16[?, ?], affine_map<(c,x0,x1)[N] -> (N,c,x1,x0)>>
        ```

        Every feature map value has a defined mapping onto a tensor value, which
        is given by the organization map. This map projects all canonical
        dimensions, including an additional "batch size" symbol, to the tensor
        space. This mapping must be surjective, i.e. every element in the result
        tensor must be defined.

        If no "batch size" symbol is provided, the corresponding dimension will
        be the naively least-varying of the output tensor, i.e. the first.
    }];

    let parameters = (ins
        DefaultValuedParameter<
                "unsigned",
                "1",
                "number of channels">:$numChannels,
        TypeParameter<"ScalarType", "element scalar type">:$scalarType,
        ArrayRefParameter<"dim_size_t", "size dimensions">:$sizes,
        OptionalParameter<"AffineMap", "tensor organization map">:$organization
    );

    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;

    let genStorageClass = 0;

    code extraClassDeclaration = [{
    /// Gets the FeatureMapDomain.
    const FeatureMapDomain &getDomain() const;
    /// Gets the lowered tensor type.
    RankedTensorType getTensorType() const;

    bool isDefaultOrganization() const
    {
        return ::mlir::dlnn::isDefaultOrganization(getOrganization());
    }
    }];
}

def DLNN_WeightsType : DLNN_Type<"Weights"> {
    let mnemonic = "weights";
    let summary = "Weights type";
    let description = [{
        A weights type is an organized type that arranges scalar values in a
        multi-dimensional shape, which has:

            - Exactly one static dimension with "input channel" semantics, `c`
            - Exactly one static dimension with "output channel" semantics, `f`
            - An arbitrary amount of static dimensions with "size" semantics
              `x_i`.

        The canonical ordering of these dimensions is `c, x_0, ..., x_R, f`,
        where `R` is the number of dimensions with "size" semantics.

        Example:
        ```mlir
        // {ins: 1, outs: 1, sclarType: f16, size: [3, 3], org: id}
        !weights<3x3xf16>
        // {ins: 3, outs: 8, sclarType: f16, size: [3, 3], org: id}
        !weights<3->3x3xf16->8>
        // {ins: 3, outs: 8, sclarType: f16, size: [3, 3], org: fchw}
        !weights<3->3x3xf16->8, affine_map<(c,x0,x1,f) -> (f,c,x1,x0)>>
        ```

        Every weights value has a defined mapping onto a tensor value, which
        is given by the organization map. This map projects all canonical
        dimensions to the tensor space.
    }];

    let parameters = (ins
        DefaultValuedParameter<
                "unsigned",
                "1",
                "number of input channels">:$numInChannels,
        DefaultValuedParameter<
                "unsigned",
                "1",
                "number of output channels">:$numOutChannels,
        TypeParameter<"ScalarType", "element scalar type">:$scalarType,
        ArrayRefParameter<"dim_size_t", "size dimensions">:$sizes,
        OptionalParameter<"AffineMap", "tensor organization map">:$organization
    );

    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;

    let genStorageClass = 0;

    code extraClassDeclaration = [{
    /// Gets the WeightsDomain.
    const WeightsDomain &getDomain() const;
    /// Gets the lowered tensor type.
    RankedTensorType getTensorType() const;

    bool isDefaultOrganization() const
    {
        return ::mlir::dlnn::isDefaultOrganization(getOrganization());
    }
    }];
}

def DLNN_OrganizedType
        : NativeTypeConcept<"OrganizedType", "organized type">;

#endif
