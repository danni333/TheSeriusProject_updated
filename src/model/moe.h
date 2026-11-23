#pragma once

#include "../moe/expert.h"
#include "../moe/load_balance.h"
#include "../moe/moe_layer.h"
#include "../moe/router.h"

namespace serius {
namespace model {

// Compatibility re-export for model/moe.h consumers.
using ::serius::model::Expert;
using ::serius::model::load_balance_loss;
using ::serius::model::MoELayer;
using ::serius::model::route;
using ::serius::model::Router;
using ::serius::model::RouterOutput;

}  // namespace model
}  // namespace serius
