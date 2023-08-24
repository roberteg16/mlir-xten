add_mlir_conversion_library(DLNNToFunc
    DLNNToFunc.cpp

    DEPENDS
        DLNNConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        DLNNIR
        MLIRFuncDialect
        MLIRIR
        MLIRPass
        MLIRTransforms
)
