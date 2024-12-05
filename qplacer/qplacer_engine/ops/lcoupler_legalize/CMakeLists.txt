set(OP_NAME lcoupler_legalize)

set(TARGET_NAME ${OP_NAME})
set(INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(LINK_LIBS qplacer_utility_cxx OpenMP::OpenMP_CXX)

add_pytorch_extension(${TARGET_NAME}_cpp 
  src/greedy_legalize.cpp
  src/legalize_bin_cpu.cpp 
  src/merge_bin_cpu.cpp 
  src/greedy_legalize_cpu.cpp 
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

install(TARGETS ${TARGET_NAME}_cpp DESTINATION qplacement/ops/${OP_NAME})

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION qplacement/ops/${OP_NAME})
