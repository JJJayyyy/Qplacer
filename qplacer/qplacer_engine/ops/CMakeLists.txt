add_subdirectory(frequency_repulsion)
add_subdirectory(utility)
add_subdirectory(greedy_legalize)
add_subdirectory(draw_place)
add_subdirectory(legality_check)

file(GLOB INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/configure.py")
install(FILES ${INSTALL_SRCS} DESTINATION qplacement/ )
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ )