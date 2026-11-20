cmake_minimum_required(VERSION 3.13)
include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)
project(threebits_int4_gru C CXX ASM)
set(CMAKE_C_STANDARD 99)
pico_sdk_init()

add_executable(threebits_int4_gru
    main.c
    ../int4_gru.c
)
target_compile_options(threebits_int4_gru PRIVATE -O3 -Wall -Wextra)
target_link_libraries(threebits_int4_gru pico_stdlib)
pico_enable_stdio_usb(threebits_int4_gru 1)
pico_enable_stdio_uart(threebits_int4_gru 0)
pico_add_extra_outputs(threebits_int4_gru)
