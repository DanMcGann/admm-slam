###########################################################################
# Find GTSAM
find_package(GTSAM REQUIRED)

###########################################################################
# Find Boost
find_package(Boost COMPONENTS serialization filesystem system program_options thread REQUIRED)

###########################################################################
# ADMM
file(GLOB srcs "src/*.cpp" "src/*.h")
include_directories(src)
add_library (admm INTERFACE ${GTSAM_LIBRARIES})
target_include_directories(admm INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(admm INTERFACE ${Boost_LIBRARIES} gtsam)

###########################################################################
# Executables
add_executable (PartitionAndRunADMM  apps/PartitionAndRunADMM)
target_link_libraries (PartitionAndRunADMM admm boost_system gtsam)

###########################################################################
#Tests
option(BUILD_ADMM_TESTS "If True builds tests for DGS" OFF)
if(${BUILD_ADMM_TESTS})
    set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${GTSAM_DIR}/../GTSAMCMakeTools")
    include(GtsamBuildTypes)
    include(GtsamTesting)
    add_subdirectory(tests)
endif()
