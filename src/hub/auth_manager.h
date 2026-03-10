#pragma once

#include <string>
#include <map>
#include <memory>
#include <jwt-cpp/jwt.h>
#include <sqlite3.h>

namespace serius::hub {

struct User {
    std::string id;
    std::string username;
    std::string email;
    std::string api_key;
    std::vector<std::string> permissions;
    std::string created_at;
    bool is_active;
};

class AuthManager {
public:
    AuthManager(const std::string& db_path, const std::string& jwt_secret);
    ~AuthManager();

    bool initialize();
    
    // Authentication
    bool authenticate_user(const std::string& token, std::string& user_id);
    bool authenticate_api_key(const std::string& api_key, std::string& user_id);
    std::string generate_jwt_token(const std::string& user_id, int expires_in_hours = 24);
    bool validate_jwt_token(const std::string& token, std::string& user_id);
    
    // User management
    bool create_user(const User& user);
    bool update_user(const std::string& user_id, const User& user);
    User get_user(const std::string& user_id);
    bool delete_user(const std::string& user_id);
    
    // Authorization
    bool check_permissions(const std::string& user_id, const std::string& action, const std::string& resource);
    bool grant_permission(const std::string& user_id, const std::string& permission);
    bool revoke_permission(const std::string& user_id, const std::string& permission);
    
    // API key management
    std::string generate_api_key(const std::string& user_id);
    bool revoke_api_key(const std::string& api_key);
    
    // Rate limiting
    bool check_rate_limit(const std::string& user_id, const std::string& endpoint);
    void record_request(const std::string& user_id, const std::string& endpoint);

private:
    sqlite3* db_;
    std::string db_path_;
    std::string jwt_secret_;
    
    // Rate limiting
    std::map<std::string, std::map<std::string, std::vector<int64_t>>> rate_limit_cache_;
    int requests_per_minute_ = 100;
    int requests_per_hour_ = 1000;
    
    bool create_tables();
    bool execute_query(const std::string& sql);
    User row_to_user(sqlite3_stmt* stmt);
    std::string generate_random_string(size_t length);
    void cleanup_expired_requests();
};

} // namespace serius::hub
