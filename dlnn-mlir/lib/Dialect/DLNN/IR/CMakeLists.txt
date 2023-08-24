add_mlir_dialect_library(DLNNIR
        Attributes.cpp
        Base.cpp
        Ops.cpp
        Types.cpp

    DEPENDS
        DLNNIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        DLNNEnums
        DLNNInterfaces
        DLNNUtils
)
