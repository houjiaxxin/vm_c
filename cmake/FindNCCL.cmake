#[=======================================================================[.rst:
FindNCCL
--------

Find the NCCL (NVIDIA Collective Communications Library) library.

Imported Targets
^^^^^^^^^^^^^^^^

``NCCL::NCCL``
  The NCCL library, if found.

Result Variables
^^^^^^^^^^^^^^^^

``NCCL_FOUND``
  True if NCCL was found.
``NCCL_INCLUDE_DIRS``
  Include directories needed to use NCCL.
``NCCL_LIBRARIES``
  Libraries needed to link to NCCL.
``NCCL_VERSION``
  Version of NCCL found.

Hints
^^^^^

``NCCL_ROOT``
``NCCL_DIR``
``NCCL_ROOT_DIR``
  Root of the NCCL installation.

``NCCL_INCLUDE_DIR``
  Include directory for NCCL headers.

``NCCL_LIBRARY``
  Path to the NCCL library file.
#]=======================================================================]

find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    HINTS
        ${NCCL_ROOT}
        ${NCCL_DIR}
        ${NCCL_ROOT_DIR}
        ENV NCCL_ROOT
        ENV NCCL_DIR
    PATH_SUFFIXES include
)

find_library(NCCL_LIBRARY
    NAMES nccl
    HINTS
        ${NCCL_ROOT}
        ${NCCL_DIR}
        ${NCCL_ROOT_DIR}
        ENV NCCL_ROOT
        ENV NCCL_DIR
    PATH_SUFFIXES lib lib64
)

if(NCCL_INCLUDE_DIR AND EXISTS "${NCCL_INCLUDE_DIR}/nccl.h")
    file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" _nccl_version_lines
        REGEX "#define[ \t]+NCCL_MAJOR[ \t]+[0-9]+|#define[ \t]+NCCL_MINOR[ \t]+[0-9]+|#define[ \t]+NCCL_PATCH[ \t]+[0-9]+"
    )
    string(REGEX REPLACE ".*NCCL_MAJOR[ \t]+([0-9]+).*" "\\1" _nccl_major "${_nccl_version_lines}")
    string(REGEX REPLACE ".*NCCL_MINOR[ \t]+([0-9]+).*" "\\1" _nccl_minor "${_nccl_version_lines}")
    string(REGEX REPLACE ".*NCCL_PATCH[ \t]+([0-9]+).*" "\\1" _nccl_patch "${_nccl_version_lines}")
    set(NCCL_VERSION "${_nccl_major}.${_nccl_minor}.${_nccl_patch}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL
    REQUIRED_VARS NCCL_LIBRARY NCCL_INCLUDE_DIR
    VERSION_VAR NCCL_VERSION
)

if(NCCL_FOUND)
    set(NCCL_INCLUDE_DIRS "${NCCL_INCLUDE_DIR}")
    set(NCCL_LIBRARIES "${NCCL_LIBRARY}")

    if(NOT TARGET NCCL::NCCL)
        add_library(NCCL::NCCL UNKNOWN IMPORTED)
        set_target_properties(NCCL::NCCL PROPERTIES
            IMPORTED_LOCATION "${NCCL_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
        )
    endif()

    mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
endif()
