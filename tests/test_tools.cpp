#include <gtest/gtest.h>

#include "tools/registry.h"
#include "tools/parser.h"

using namespace serius;
using namespace serius::tools;

TEST(ToolRegistry, RegisterTool)
{
    ToolRegistry registry;

    ToolDefinition tool("test_tool", "A test tool", json{},
                         [](const json& args) -> json {
                             return json{{"result", "success"}};
                         });

    registry.register_tool(tool);

    EXPECT_TRUE(registry.has_tool("test_tool"));
    EXPECT_EQ(registry.list_tools().size(), 1);
}

TEST(ToolRegistry, ExecuteTool)
{
    ToolRegistry registry;

    json schema;
    schema["type"] = "object";
    schema["properties"]["a"]["type"] = "number";
    schema["properties"]["b"]["type"] = "number";

    ToolDefinition tool("add", "Add two numbers", schema,
                         [](const json& args) -> json {
                             double a = args["a"];
                             double b = args["b"];
                             return json{{"result", a + b}};
                         });

    registry.register_tool(tool);

    json args;
    args["a"] = 3;
    args["b"] = 5;
    json result = registry.execute("add", args);

    EXPECT_EQ(result["result"], 8);
}

TEST(ToolRegistry, ExecuteInvalidTool)
{
    ToolRegistry registry;

    json args = json{};
    json result = registry.execute("nonexistent", args);

    EXPECT_TRUE(result.contains("error"));
}

TEST(ToolRegistry, ExecuteInvalidArguments)
{
    ToolRegistry registry;

    json schema;
    schema["type"] = "object";
    schema["properties"]["required_field"]["type"] = "string";

    ToolDefinition tool("test", "Test tool", schema,
                         [](const json& args) -> json {
                             return json{{"result", "success"}};
                         });

    registry.register_tool(tool);

    json args = json{};  // Missing required field
    json result = registry.execute("test", args);

    EXPECT_TRUE(result.contains("error"));
}

TEST(ToolRegistry, UnregisterTool)
{
    ToolRegistry registry;

    ToolDefinition tool("test", "Test", json{},
                         [](const json&) -> json { return json{}; });

    registry.register_tool(tool);
    EXPECT_TRUE(registry.has_tool("test"));

    registry.unregister_tool("test");
    EXPECT_FALSE(registry.has_tool("test"));
}

TEST(ToolRegistry, ListTools)
{
    ToolRegistry registry;

    ToolDefinition tool1("tool1", "Tool 1", json{},
                         [](const json&) -> json { return json{}; });
    ToolDefinition tool2("tool2", "Tool 2", json{},
                         [](const json&) -> json { return json{}; });

    registry.register_tool(tool1);
    registry.register_tool(tool2);

    auto names = registry.list_tools();
    EXPECT_EQ(names.size(), 2);
}

TEST(ToolRegistry, Clear)
{
    ToolRegistry registry;

    ToolDefinition tool("test", "Test", json{},
                         [](const json&) -> json { return json{}; });

    registry.register_tool(tool);
    EXPECT_EQ(registry.list_tools().size(), 1);

    registry.clear();
    EXPECT_EQ(registry.list_tools().size(), 0);
}

TEST(FunctionCallParser, HasFunctionCall)
{
    FunctionCallParser parser;

    std::string text1 = "This is normal text";
    std::string text2 = "{\"function\": \"test\", \"arguments\": {\"x\": 1}}";

    EXPECT_FALSE(parser.has_function_call(text1));
    EXPECT_TRUE(parser.has_function_call(text2));
}

TEST(FunctionCallParser, ParseFunctionCall)
{
    FunctionCallParser parser;

    std::string text = "{\"function\": \"weather.get\", \"arguments\": {\"city\": \"Rome\"}}";

    auto call = parser.parse(text);

    ASSERT_TRUE(call.has_value());
    EXPECT_EQ(call->name, "weather.get");
    EXPECT_EQ(call->arguments["city"], "Rome");
}

TEST(FunctionCallParser, ParseInvalidJSON)
{
    FunctionCallParser parser;

    std::string text = "This is not JSON";

    auto call = parser.parse(text);

    EXPECT_FALSE(call.has_value());
}

TEST(FunctionCallParser, ParseMissingFields)
{
    FunctionCallParser parser;

    std::string text = "{\"function\": \"test\"}";  // Missing arguments

    auto call = parser.parse(text);

    EXPECT_FALSE(call.has_value());
}

TEST(FunctionCallParser, ParseInText)
{
    FunctionCallParser parser;

    std::string text = "Let me check the weather for you. "
                      "{\"function\": \"weather.get\", \"arguments\": {\"city\": \"Milan\"}} "
                      "It will be sunny.";

    auto call = parser.parse(text);

    ASSERT_TRUE(call.has_value());
    EXPECT_EQ(call->name, "weather.get");
}
