if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

unset(Q_IMPLANT_TESTS)

macro(addeval NAME)
  list(APPEND Q_IMPLANT_TESTS ${NAME})
endmacro(addeval)

include("test.lst")

set(TFLITE2CIRCLE_PATH $<TARGET_FILE:tflite2circle>)

nncc_find_resource(PyTorchExamples)
set(Torch_RECIPE_REPO "${PyTorchExamples_DIR}")

add_test(NAME q_implant_export_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/temp.sh"
          "${CMAKE_CURRENT_BINARY_DIR}"
          "${Torch_RECIPE_REPO}"
          "${NNCC_OVERLAY_DIR}/venv_2_12_1"
          "${TFLITE2CIRCLE_PATH}"
          ${Q_IMPLANT_TESTS}
)