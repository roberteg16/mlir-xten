add_mlir_dialect_library(DLNNInterfaces
        EnclaveOpInterface.cpp
        GraphInterface.cpp
        NodeInterface.cpp
        ScalarInterface.cpp
        VolumeInterface.cpp
        VolumeOpInterface.cpp

    DEPENDS
        DLNNIncGen

    LINK_LIBS PUBLIC
        MLIRIR
)
