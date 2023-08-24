################################################################################
# DLNNIncGen
#
# MLIR DLNN dialect generated includes.
################################################################################

add_custom_target(DLNNIncGen)

mlir_gen_enums(DLNN)

# Interfaces.
add_subdirectory(Interfaces)
# Attributes, Dialect, Operations and Types.
add_subdirectory(IR)
# Passes.
add_subdirectory(Transforms)
