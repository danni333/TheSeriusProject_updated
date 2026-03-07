#include <gtest/gtest.h>

namespace serius {
namespace core {
int dummy_function();
}
}  // namespace serius

TEST(CoreDummy, Returns42)
{
    EXPECT_EQ(serius::core::dummy_function(), 42);
}
