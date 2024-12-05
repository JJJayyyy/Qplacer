
set(OPENBLAS OFF CACHE BOOL "Turn off OPENBLAS for Limbo")
set(INSTALL_LIMBO OFF CACHE BOOL "Turn off installation for Limbo")
set(GENERATE_DOCS OFF CACHE BOOL "Turn off documentation for Limbo")
set(ENABLE_TEST OFF CACHE BOOL "Turn off test for Limbo")

add_subdirectory(Limbo_qplacer_backup)

if(INSTALL_BACKUP)
    set(DREAMPLACE_OPERATIONS_BASE "DREAMPlace_qplacer_backup/dreamplace/ops")
else()
    set(DREAMPLACE_OPERATIONS_BASE "DREAMPlace/dreamplace/ops")
endif()

set(SUBDIRECTORIES
    utility
    place_io
    dct
    move_boundary
    electric_potential
    hpwl
    logsumexp_wirelength
    pin_pos
    pin_weight_sum
    weighted_average_wirelength
)

foreach(DIR IN LISTS SUBDIRECTORIES)
    add_subdirectory("${DREAMPLACE_OPERATIONS_BASE}/${DIR}")
endforeach()