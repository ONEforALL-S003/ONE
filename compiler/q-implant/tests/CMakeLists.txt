if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

unset(Q_IMPLANT_TESTS)

# for value test
nnas_find_package(Jsoncpp)
if(NOT Jsoncpp_FOUND)
  message(STATUS "q-implant-value-test: Jsoncpp NOT FOUND")
  return()
endif(NOT Jsoncpp_FOUND)

nnas_find_package(LibnpySource QUIET)
if(NOT LibnpySource_FOUND)
  message(STATUS "q-implant-value-test: LibnpySource NOT FOUND")
  return()
endif(NOT LibnpySource_FOUND)

set(DRIVER "src/TestDriver.cpp")

add_executable(test-driver ${DRIVER})
target_include_directories(test-driver PRIVATE ${Jsoncpp_INCLUDE_DIRS})
target_include_directories(test-driver PRIVATE ${LibnpySource_DIR}/include)
target_include_directories(test-driver PRIVATE include)

target_link_libraries(test-driver ${Jsoncpp_STATIC_LIB})
target_link_libraries(test-driver safemain)
target_link_libraries(test-driver loco)
target_link_libraries(test-driver luci_import)
target_link_libraries(test-driver luci_service)
target_link_libraries(test-driver luci_pass)
target_link_libraries(test-driver luci_export)
target_link_libraries(test-driver luci_env)
target_link_libraries(test-driver arser)
target_link_libraries(test-driver vconone)

unset(Q_IMPLANT_VALUE_TESTS)
# for value test

macro(addeval NAME)
  list(APPEND Q_IMPLANT_TESTS ${NAME})
endmacro(addeval)

include("test.lst")

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

add_test(NAME q_implant_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/q-implant.sh"
          "${CMAKE_CURRENT_BINARY_DIR}"
          "${ARTIFACTS_BIN_PATH}"
          "${NNCC_OVERLAY_DIR}/venv_2_12_1"
          "$<TARGET_FILE:q-implant>"
          ${Q_IMPLANT_TESTS}
)

add_test(NAME q_implant_op_level_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/q-implant-op-level-value-test.sh"
          "${CMAKE_CURRENT_BINARY_DIR}"
          "${ARTIFACTS_BIN_PATH}"
          ${Q_IMPLANT_TESTS}
)
