# Hub components for Community Models Hub (Phase 65)

# SQLite amalgamation: fetched at top level as sqlite-autoconf (no
# CMakeLists.txt inside), so compile the single-file amalgamation directly.
set(SERIUS_HUB_SQLITE OFF)
if(EXISTS "${sqlite3_SOURCE_DIR}/sqlite3.c")
    add_library(serius_sqlite3 STATIC ${sqlite3_SOURCE_DIR}/sqlite3.c)
    target_include_directories(serius_sqlite3 PUBLIC ${sqlite3_SOURCE_DIR})
    target_compile_definitions(serius_sqlite3 PRIVATE
        SQLITE_THREADSAFE=1
        SQLITE_OMIT_LOAD_EXTENSION)
    if(MSVC)
        # sqlite3.c is third-party C; keep the build log clean.
        set_source_files_properties(${sqlite3_SOURCE_DIR}/sqlite3.c PROPERTIES
            COMPILE_OPTIONS "/W0")
    endif()
    set(SERIUS_HUB_SQLITE ON)
else()
    message(WARNING "sqlite3 amalgamation not found (${sqlite3_SOURCE_DIR}); "
                    "hub metadata will use the JSON file backend")
endif()

add_library(hub STATIC
    model_hub_server.cpp
    metadata_manager.cpp
    # auth_manager.cpp  # Requires OpenSSL/JWT - disabled for now
    plugin_manager.cpp
    search_engine.cpp
    nomic_bert.cpp
    converter.cpp
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
    # OpenSSL::SSL  # Requires OpenSSL - disabled for now
    # OpenSSL::Crypto
    # jwt-cpp  # Requires JWT - disabled for now
)

if(SERIUS_HUB_SQLITE)
    target_link_libraries(hub serius_sqlite3)
    # PUBLIC: consumers of metadata_manager.h (tests, hub server) must see the
    # same class layout — the sqlite3* member is guarded by this define.
    target_compile_definitions(hub PUBLIC SERIUS_HUB_SQLITE=1)
endif()

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
