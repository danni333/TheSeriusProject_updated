#pragma once

#include "../infer/speculative.h"

namespace serius {
namespace inference {

class DraftModel : public infer::DraftModel
{
public:
    using infer::DraftModel::DraftModel;
};

class SpeculativeDecoder : public infer::SpeculativeDecoder
{
public:
    using infer::SpeculativeDecoder::SpeculativeDecoder;
};

}  // namespace inference
}  // namespace serius
