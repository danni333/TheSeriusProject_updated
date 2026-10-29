#pragma once

#include <httplib.h>
#include <string>

namespace serius {
namespace server {

/// Setup static file serving routes for the web UI.
/// Serves files from the ui/ directory.
void setup_static_routes(httplib::Server &server, const std::string &ui_dir = "ui");

}  // namespace server
}  // namespace serius

