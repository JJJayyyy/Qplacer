
set(OPENBLAS OFF CACHE BOOL "Turn off OPENBLAS for Limbo")
set(INSTALL_LIMBO OFF CACHE BOOL "Turn off installation for Limbo")
set(GENERATE_DOCS OFF CACHE BOOL "Turn off documentation for Limbo")
set(ENABLE_TEST OFF CACHE BOOL "Turn off test for Limbo")


option(INSTALL_BACKUP "Enable qplacer backup thridparty" ON)

if(INSTALL_BACKUP)
    add_subdirectory(Limbo_qplacer_backup)
    set(DREAMPLACE_OPERATIONS_BASE "DREAMPlace_qplacer_backup/dreamplace/ops")
else()
    add_subdirectory(Limbo)
    set(DREAMPLACE_OPERATIONS_BASE "DREAMPlace/dreamplace/ops")
endif()

set(SUBDIRECTORIES
    utility
    place_io
    dct
    pin_pos
    move_boundary
    electric_potential
    hpwl
    weighted_average_wirelength
    logsumexp_wirelength
    pin_weight_sum
    macro_legalize
    greedy_legalize
    abacus_legalize
)

foreach(DIR IN LISTS SUBDIRECTORIES)
    add_subdirectory("${DREAMPLACE_OPERATIONS_BASE}/${DIR}")
endforeach()