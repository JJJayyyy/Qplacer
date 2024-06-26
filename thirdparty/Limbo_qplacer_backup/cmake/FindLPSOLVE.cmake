##############################################################################
## From the Ball Project http://www.ball-project.org/                       ##
## https://bitbucket.org/ball/ball/src/b4bdcec12ee5/cmake/FindLPSolve.cmake ##
##                                                                          ##
## Some modifications have been done here to make it work better on Windows ##
##############################################################################

## Detect lpsolve
INCLUDE(CheckCXXSourceCompiles)

SET(LPSOLVE_INCLUDE_DIRS "" CACHE STRING "Full path to the lpsolve headers")
MARK_AS_ADVANCED(LPSOLVE_INCLUDE_DIRS)

SET(LPSOLVE_LIBRARIES "" CACHE STRING "Full path to the lpsolve55 library (including the library)")
MARK_AS_ADVANCED(LPSOLVE_LIBRARIES)

SET(LPSOLVE_INCLUDE_TRIAL_PATH
    /sw/include
    /usr/include
    /usr/local/include
    /opt/include
    /opt/local/include
    "$ENV{LPSOLVE_DIR}"
    "$ENV{LPSOLVE_HOME}"
    )

FIND_PATH(LPSOLVE_INCLUDE_PATH
    NAMES lpsolve/lp_lib.h lp_lib.h
    PATHS ${LPSOLVE_INCLUDE_PATH} ${LPSOLVE_INCLUDE_TRIAL_PATH}
    )
IF (LPSOLVE_INCLUDE_PATH)
    STRING(REGEX REPLACE "/lp_lib.h$" "" LPSOLVE_INCLUDE_PATH ${LPSOLVE_INCLUDE_PATH})
    SET(LPSOLVE_INCLUDE_DIRS ${LPSOLVE_INCLUDE_PATH} CACHE STRING "Full path to the lpsolve headers" FORCE)

    GET_FILENAME_COMPONENT(LPSOLVE_INSTALL_BASE_PATH ${LPSOLVE_INCLUDE_DIRS} PATH)

    SET(LPSOLVE_LIB_TRIALPATH
        ${LPSOLVE_INCLUDE_DIRS}
        ${LPSOLVE_INSTALL_BASE_PATH}/lib
        /usr/lib/
        /usr/local/lib
        /opt/lib
        )
    # add both paths to include directories, as we are not sure whehter there is an lpsolve subdirectory 
    SET(LPSOLVE_INCLUDE_DIRS ${LPSOLVE_INCLUDE_DIRS};${LPSOLVE_INCLUDE_DIRS}/lpsolve)

    # hints of names from version 55 to 99
    set(LPSOLVE_NAME_HINTS "")
    foreach(LPSOLVE_VERSION RANGE 55 99)
        list(APPEND LPSOLVE_NAME_HINTS "lpsolve${LPSOLVE_VERSION}")
    endforeach()

    FIND_LIBRARY(LOCAL_LPSOLVE_LIBRARIES
        NAMES ${LPSOLVE_NAME_HINTS}
        PATHS ${LPSOLVE_LIBRARIES} ${LPSOLVE_LIB_TRIALPATH}
        )
    SET(LPSOLVE_LIBRARIES ${LOCAL_LPSOLVE_LIBRARIES} CACHE STRING "Full path to the lpsolve55 library (including the library)" FORCE)
    IF (LPSOLVE_LIBRARIES)
        SET(LPSOLVE_FOUND TRUE)

        ## Try to find out if lpsolve can link standalone
        SET(LPSOLVE_TRY_CODE
            "#include <lp_lib.h>
            int main(int /*argc*/, char** /*argv*/) {
            int major, minor, release, build;

            lp_solve_version(&major, &minor, &release, &build);

            return 0;
            }")

        SET(CMAKE_REQUIRED_INCLUDES ${LPSOLVE_INCLUDE_DIRS})
        SET(CMAKE_REQUIRED_LIBRARIES ${LPSOLVE_LIBRARIES})
        CHECK_CXX_SOURCE_COMPILES("${LPSOLVE_TRY_CODE}" LPSOLVE_LINKS_ALONE)
        SET(CMAKE_REQUIRED_LIBRARIES "")
        SET(CMAKE_REQUIRED_INCLUDES "")

        ## Try to find out if lpsolve can link with some extra libs
        IF (NOT LPSOLVE_LINKS_ALONE)
            FIND_LIBRARY(LPSOLVE_LIB_DL "dl")
            FIND_LIBRARY(LPSOLVE_LIB_COLAMD "colamd")

            LIST(APPEND LPSOLVE_LIBRARIES "${LPSOLVE_LIB_DL}" "${LPSOLVE_LIB_COLAMD}")

            SET(CMAKE_REQUIRED_INCLUDES ${LPSOLVE_INCLUDE_DIRS})
            SET(CMAKE_REQUIRED_LIBRARIES ${LPSOLVE_LIBRARIES})
            CHECK_CXX_SOURCE_COMPILES("${LPSOLVE_TRY_CODE}" LPSOLVE_LINKS_WITH_EXTRA_LIBS)
            SET(CMAKE_REQUIRED_LIBRARIES "")
            SET(CMAKE_REQUIRED_INCLUDES "")

            IF (NOT LPSOLVE_LINKS_WITH_EXTRA_LIBS)
                MESSAGE(STATUS "Could not link against lpsolve55!")
            ENDIF(NOT LPSOLVE_LINKS_WITH_EXTRA_LIBS)
        ENDIF(NOT LPSOLVE_LINKS_ALONE)
    ENDIF(LPSOLVE_LIBRARIES)
ENDIF(LPSOLVE_INCLUDE_PATH)

IF (LPSOLVE_LINKS_ALONE OR LPSOLVE_LINKS_WITH_EXTRA_LIBS)
    SET(LPSOLVE_LINKS TRUE)
ENDIF(LPSOLVE_LINKS_ALONE OR LPSOLVE_LINKS_WITH_EXTRA_LIBS)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LPSOLVE DEFAULT_MSG
    LPSOLVE_LIBRARIES
    LPSOLVE_INCLUDE_DIRS
    LPSOLVE_LINKS
    )
