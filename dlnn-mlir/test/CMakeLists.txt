# TODO: Add license.

# ###############################################################################
# check-dlnn-mlir
#
# The dlnn-mlir regression test project.
# ###############################################################################

project(check-dlnn-mlir)

# Configure the testing site configuration.
configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)
set(TEST_DEPENDS
    FileCheck count not
    dlnn-opt
)

# Create the test suite.
add_lit_testsuite(${PROJECT_NAME}
    "Running the dlnn-mlir regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
    ${TEST_DEPENDS}
)
add_lit_testsuites(DLNNMLIR
    ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS
    ${TEST_DEPENDS}
)
