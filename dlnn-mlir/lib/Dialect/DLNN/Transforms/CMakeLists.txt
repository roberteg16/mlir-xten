add_mlir_dialect_library(DLNNTransforms
        ApplyTiling.cpp
        Flatten.cpp
        Simplify.cpp

    DEPENDS
        DLNNIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        MLIRTransforms
        DLNNIR
)
