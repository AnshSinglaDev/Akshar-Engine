cmake_minimum_required(VERSION 3.20)
project(akshar_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED LLAMA_ROOT)
    set(LLAMA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../hinglish_research/llama.cpp" CACHE PATH "Path to llama.cpp root")
endif()
if(NOT DEFINED ORT_GPU_ROOT)
    set(ORT_GPU_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../ort_gpu_win" CACHE PATH "Path to ONNX Runtime native runtimes")
endif()
if(NOT DEFINED ORT_INC_ROOT)
    set(ORT_INC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../ort_dml" CACHE PATH "Path to ONNX Runtime headers")
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LLAMA_ROOT}/include
    ${LLAMA_ROOT}/ggml/include
    ${ORT_INC_ROOT}/build/native/include
)

link_directories(
    ${LLAMA_ROOT}/build/src/Release
    ${LLAMA_ROOT}/build/ggml/src/Release
    ${ORT_GPU_ROOT}/runtimes/win-x64/native
)

add_executable(akshar_engine
    src/main.cpp
    src/engine.cpp
    src/onnx_engine.cpp
)

target_link_libraries(akshar_engine PRIVATE
    llama
    ggml
    ggml-base
    ggml-cpu
    onnxruntime
)

if (MSVC)
    target_compile_definitions(akshar_engine PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(akshar_engine PRIVATE
        /O2
        /fp:fast
        /arch:AVX2
        /W3
    )
    add_custom_command(TARGET akshar_engine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LLAMA_ROOT}/build/bin/Release/llama.dll"
        $<TARGET_FILE_DIR:akshar_engine>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LLAMA_ROOT}/build/bin/Release/ggml.dll"
        $<TARGET_FILE_DIR:akshar_engine>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LLAMA_ROOT}/build/bin/Release/ggml-base.dll"
        $<TARGET_FILE_DIR:akshar_engine>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LLAMA_ROOT}/build/bin/Release/ggml-cpu.dll"
        $<TARGET_FILE_DIR:akshar_engine>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LLAMA_ROOT}/build/bin/Release/ggml-vulkan.dll"
        $<TARGET_FILE_DIR:akshar_engine>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_GPU_ROOT}/runtimes/win-x64/native/onnxruntime.dll"
        $<TARGET_FILE_DIR:akshar_engine>
    )
else()
    target_compile_options(akshar_engine PRIVATE
        -O3
        -ffast-math
        -mavx2
        -Wall
    )
endif()
