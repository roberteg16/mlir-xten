//===- VolumeOpInterface.td - DLNN volume op interface -----*- tablegen -*-===//
//
// This is the definitions file for the DLNN volume op interface.
//
//===----------------------------------------------------------------------===//

#ifndef DLNN_VOLUMEOPINTERFACE
#define DLNN_VOLUMEOPINTERFACE

include "mlir/IR/OpBase.td"

def DLNN_VolumeOpInterface : OpInterface<"VolumeOpInterface"> {
    let cppNamespace = "::mlir::dlnn";

    let description = [{
        Interface for an operation that implements a comprehension on volumes.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Obtains a relation between volume elements.

                Given a source volume to construct a virtual iteration space
                over, this method returns a map from all elements of this space
                (and thus the source volume) to all dependent elements of all
                other volumes involved in this comprehension.

                It is required that the virtual iteration space is constructed
                such that the leading dimensions are identity mappings onto the
                source volume, and the trailing dimensions are arbitrary but
                fixed assignments for the reductions in the comprehension, if
                any.
            }],
            /*retTy=*/"std::optional<VolumeRelation>",
            /*methodName=*/"getVolumeRelation",
            /*args=*/(ins "OpTiedValue":$source),
            /*methodBody=*/"",
            /*defaultImplementation=*/""
        >
    ];
}

#endif
