//===- VolumeInterface.td - DLNN volume interface ----------*- tablegen -*-===//
//
// This is the definitions file for the DLNN volume interface.
//
//===----------------------------------------------------------------------===//

#ifndef DLNN_VOLUMETYPEINTERFACE
#define DLNN_VOLUMETYPEINTERFACE

include "mlir/IR/OpBase.td"

def DLNN_VolumeInterface : TypeInterface<"VolumeInterface"> {
    let cppNamespace = "::mlir::dlnn";

    let description = [{
        Interface for a type that is a multidimensionally indexed container.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Gets the set of valid indices for a volume instance.
            }],
            /*retTy=*/"std::optional<SetAndSymbolValues>",
            /*methodName=*/"getIndexDomain",
            /*args=*/(ins "OpBuilder &":$builder, "Value":$value),
            /*methodBody=*/"",
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/[{Gets the type of the specified subvolume.}],
            /*retTy=*/"Type",
            /*methodName=*/"getSubvolumeType",
            /*args=*/(ins "ArrayRef<int64_t>":$shape),
            /*methodBody=*/"",
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/[{Obtain an undefined volume.}],
            /*retTy=*/"Value",
            /*methodName=*/"createUndefinedVolume",
            /*args=*/(ins
                "OpBuilder &":$builder,
                "ArrayRef<OpFoldResult>":$sizes,
                "Location":$location),
            /*methodBody=*/"",
            /*defaultImplementation=*/""
        >,
        InterfaceMethod<
            /*desc=*/[{Extracts a subvolume from a volume instance.}],
            /*retTy=*/"Value",
            /*methodName=*/"extractSubvolume",
            /*args=*/(ins
                "OpBuilder &":$builder,
                "Value":$volume,
                "const HyperRectangle &":$region),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{ return Value{}; }]
        >,
        InterfaceMethod<
            /*desc=*/[{Inserts a subvolume into a volume instance.}],
            /*retTy=*/"Value",
            /*methodName=*/"insertSubvolume",
            /*args=*/(ins
                "OpBuilder &":$builder,
                "Value":$volume,
                "const HyperRectangle &":$region,
                "Value":$subvolume),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{ return Value{}; }]
        >
    ];

    code extraSharedClassDeclaration = [{
    Type getSubvolumeType(const HyperRectangle &region) const
    {
        return $_type.getSubvolumeType(region.makeShape());
    }
    }];
}

#endif
