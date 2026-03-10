#include "safety/guardrails.h"
#include <gtest/gtest.h>

TEST(Guardrails, BlocklistToken)
{
    serius::safety::Guardrails g;
    g.add_blocklist_token(42);
    EXPECT_TRUE(g.is_blocked_token(42));
    EXPECT_FALSE(g.is_blocked_token(0));
    g.remove_blocklist_token(42);
    EXPECT_FALSE(g.is_blocked_token(42));
}

TEST(Guardrails, CheckResponse)
{
    serius::safety::Guardrails g;
    g.add_blocklist_phrase("badword");
    EXPECT_FALSE(g.check_response("this has badword inside"));
    EXPECT_TRUE(g.check_response("clean text"));
    g.clear();
    EXPECT_TRUE(g.check_response("this has badword inside"));
}
