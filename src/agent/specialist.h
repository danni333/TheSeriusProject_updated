#pragma once

#include <string>

#include "model_interface.h"

namespace serius {
namespace agent {

/// A specialized agent that handles a specific task type.
class Specialist
{
   public:
    Specialist(std::string name, std::string specialty, Model *model);

    const std::string &name() const { return name_; }
    const std::string &specialty() const { return specialty_; }

    /// Execute a task using this specialist's model.
    std::string execute(const std::string &task) const;

    /// Check if this specialist can handle a task keyword.
    bool can_handle(const std::string &keyword) const;

   private:
    std::string name_;
    std::string specialty_;
    Model *model_;  // Not owned
};

}  // namespace agent
}  // namespace serius