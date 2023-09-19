if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

unset(Q_IMPLANT_TESTS)

macro(addeval NAME)
  list(APPEND Q_IMPLANT_TESTS ${NAME})
endmacro(addeval)

include("test.lst")

set(TFLITE2CIRCLE_PATH $<TARGET_FILE:tflite2circle>)

get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

add_test(NAME q_implant_export_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/temp.sh"
          "${CMAKE_CURRENT_BINARY_DIR}"
          "${ARTIFACTS_BIN_PATH}"
          "${NNCC_OVERLAY_DIR}/venv_2_12_1"
          "${TFLITE2CIRCLE_PATH}"
          ${Q_IMPLANT_TESTS}
)