FUNCTION(ADD_EXAMPLE_MAIN target)
    GET_FILENAME_COMPONENT(name ${target} NAME_WE)
    STRING(REPLACE "_" "-" name ${name})
    ADD_EXECUTABLE(${name} ${target})

    IF(USE_OPENBLAS)
        TARGET_USE_BLAS(${name})
    ENDIF()

    TARGET_USE_STDTENSOR(${name})
    TARGET_USE_STDNN_OPS(${name})
    TARGET_USE_STDTRACER(${name})
ENDFUNCTION()
