
add_subdirectory(utility)
add_subdirectory(draw_place)
add_subdirectory(frequency_repulsion)
add_subdirectory(lcoupler_legalize)
add_subdirectory(qubit_legalize)
add_subdirectory(legality_check)

file(GLOB INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/configure.py")
install(FILES ${INSTALL_SRCS} DESTINATION qplacement/ )
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ )