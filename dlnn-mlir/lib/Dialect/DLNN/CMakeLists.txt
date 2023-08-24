add_subdirectory(Interfaces)
add_subdirectory(IR)
add_subdirectory(Transforms)
add_subdirectory(Utils)

add_mlir_dialect_library(DLNNEnums
        Enums.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
)
