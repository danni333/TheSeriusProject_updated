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
    schema["required"] = {"required_field"};

    ToolDefinition tool("test", "Test tool", schema,
                         [](const json& args) -> json {
                             return json{{"result", "success"}};
                         });

    registry.register_tool(tool);

    json args = json{};  // Missing required field
    json result = registry.execute("test", args);

    EXPECT_TRUE(result.contains("error"));
}

TEST(ToolRegistry, OptionalPropertiesAreNotRequired)
{
    ToolRegistry registry;

    json schema;
    schema["type"] = "object";
    schema["properties"]["optional_field"]["type"] = "string";
    schema["properties"]["other"]["type"] = "integer";
    schema["required"] = {};  // nothing required

    ToolDefinition tool("test", "Test tool", schema,
                         [](const json&) -> json {
                             return json{{"result", "ok"}};
                         });
    registry.register_tool(tool);

    // Empty args must pass: no field is declared required.
    json result = registry.execute("test", json{});
    EXPECT_FALSE(result.contains("error"));
    EXPECT_EQ(result["result"], "ok");
}

TEST(ToolRegistry, WrongArgumentTypeIsRejected)
{
    ToolRegistry registry;

    json schema;
    schema["type"] = "object";
    schema["properties"]["count"]["type"] = "integer";
    schema["required"] = {"count"};

    ToolDefinition tool("test", "Test tool", schema,
                         [](const json&) -> json {
                             return json{{"result", "ok"}};
                         });
    registry.register_tool(tool);

    json bad = json{{"count", "not a number"}};
    EXPECT_TRUE(registry.execute("test", bad).contains("error"));

    json good = json{{"count", 3}};
    EXPECT_FALSE(registry.execute("test", good).contains("error"));
}

TEST(ToolRegistry, SchemaEnumAndRangeValidation)
{
    ToolRegistry registry;

    json schema;
    schema["type"] = "object";
    schema["properties"]["color"]["type"] = "string";
    schema["properties"]["color"]["enum"] = {"red", "green", "blue"};
    schema["properties"]["count"]["type"] = "integer";
    schema["properties"]["count"]["minimum"] = 1;
    schema["properties"]["count"]["maximum"] = 10;

    ToolDefinition tool("test", "Test", schema,
                        [](const json&) -> json { return json{{"ok", true}}; });
    registry.register_tool(tool);

    EXPECT_TRUE(registry.execute("test", json{{"color", "red"}, {"count", 5}}).contains("ok"));
    EXPECT_TRUE(registry.execute("test", json{{"color", "purple"}}).contains("error"));
    EXPECT_TRUE(registry.execute("test", json{{"count", 0}}).contains("error"));
    EXPECT_TRUE(registry.execute("test", json{{"count", 11}}).contains("error"));
}

TEST(ToolRegistry, SchemaStringAndArrayConstraints)
{
    ToolRegistry registry;

    json schema;
    schema["type"] = "object";
    schema["properties"]["name"]["type"] = "string";
    schema["properties"]["name"]["minLength"] = 3;
    schema["properties"]["name"]["maxLength"] = 8;
    schema["properties"]["name"]["pattern"] = "^[a-z]+$";
    schema["properties"]["tags"]["type"] = "array";
    schema["properties"]["tags"]["items"]["type"] = "integer";
    schema["properties"]["tags"]["minItems"] = 1;
    schema["properties"]["tags"]["maxItems"] = 3;

    ToolDefinition tool("test", "Test", schema,
                        [](const json&) -> json { return json{{"ok", true}}; });
    registry.register_tool(tool);

    EXPECT_TRUE(registry.execute("test", json{{"name", "abc"}, {"tags", {1, 2}}}).contains("ok"));
    EXPECT_TRUE(registry.execute("test", json{{"name", "ab"}}).contains("error"));      // too short
    EXPECT_TRUE(registry.execute("test", json{{"name", "abcdefghi"}}).contains("error")); // too long
    EXPECT_TRUE(registry.execute("test", json{{"name", "ABC"}}).contains("error"));       // pattern
    EXPECT_TRUE(registry.execute("test", json{{"tags", {1, 2, 3, 4}}}).contains("error")); // too many
    EXPECT_TRUE(registry.execute("test", json{{"tags", {"x"}}}).contains("error"));       // bad item type
    EXPECT_TRUE(registry.execute("test", json{{"tags", json::array()}}).contains("error")); // too few
}

TEST(ToolRegistry, SchemaAdditionalPropertiesAndNestedObject)
{
    ToolRegistry registry;

    json schema;
    schema["type"] = "object";
    schema["properties"]["point"]["type"] = "object";
    schema["properties"]["point"]["properties"]["x"]["type"] = "number";
    schema["properties"]["point"]["properties"]["y"]["type"] = "number";
    schema["properties"]["point"]["required"] = {"x", "y"};
    schema["properties"]["point"]["additionalProperties"] = false;

    ToolDefinition tool("test", "Test", schema,
                        [](const json&) -> json { return json{{"ok", true}}; });
    registry.register_tool(tool);

    EXPECT_TRUE(registry.execute("test", json{{"point", {{"x", 1.0}, {"y", 2.0}}}}).contains("ok"));
    EXPECT_TRUE(registry.execute("test", json{{"point", {{"x", 1.0}}}}).contains("error"));  // missing y
    EXPECT_TRUE(registry.execute("test", json{{"point", {{"x", 1.0}, {"y", 2.0}, {"z", 3.0}}}})
                    .contains("error"));  // closed object
    EXPECT_TRUE(registry.execute("test", json{{"point", {{"x", "nan"}, {"y", 2.0}}}})
                    .contains("error"));  // wrong nested type
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
