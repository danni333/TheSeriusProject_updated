# Hub components for Community Models Hub (Phase 65)

add_library(hub STATIC
    model_hub_server.cpp
    metadata_manager.cpp
    # auth_manager.cpp  # Requires OpenSSL/JWT - disabled for now
    plugin_manager.cpp
    search_engine.cpp
)

target_include_directories(hub PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(hub
    serius_core
    serius_server
    httplib::httplib
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    # sqlite3
    # OpenSSL::SSL  # Requires OpenSSL - disabled for now
    # OpenSSL::Crypto
    # jwt-cpp  # Requires JWT - disabled for now
)

# Platform-specific libraries
if(WIN32)
    target_link_libraries(hub
        psapi
    )
endif()

# Hub executable (optional standalone server)
add_executable(serius_hub_server
    hub_main.cpp
)

target_link_libraries(serius_hub_server
    hub
    serius_core
    serius_server
)

# Install hub components
install(TARGETS hub serius_hub_server
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    model_hub_server.h
    metadata_manager.h
    # auth_manager.h  # Requires OpenSSL/JWT - disabled for now
    plugin_interface.h
    plugin_manager.h
    search_engine.h
    DESTINATION include/serius/hub
)
