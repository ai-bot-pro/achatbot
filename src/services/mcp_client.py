from dataclasses import dataclass
import logging
import json
import os
import traceback
from typing import Any, Dict, List, Optional, Union

from src.processors.llm.base import LLMProcessor
from src.schemas.function_schema import FunctionSchema
from src.schemas.tools_schema import ToolsSchema
from src.common.event import EventHandlerManager
from src.types.ai_conf import MCPServerConfig

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.session_group import SseServerParameters, StreamableHttpParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    from mcp.client.streamable_http import streamablehttp_client
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error("In order to use an MCP client, you need to `pip install achatbot[mcp]`.")
    raise Exception(f"Missing module: {e}")

# NOTE: for short session case, list_tools and call_tool with client session wrap, not long session


class MultiMCPClients(EventHandlerManager):
    def __init__(
        self,
        mcp_servers_config: Dict[str, MCPServerConfig],
    ):
        super().__init__()
        self.regist_clients: Dict[str, MCPClient] = {}
        for name, config in mcp_servers_config.items():
            if config.transport == "stdio":
                server_params = StdioServerParameters(**config.parameters)
                server_params.env = (
                    {**os.environ, **server_params.env} if server_params.env else {**os.environ}
                )
            elif config.transport == "sse":
                server_params = SseServerParameters(**config.parameters)
            elif config.transport == "streamable-http":
                server_params = StreamableHttpParameters(**config.parameters)
            else:
                logging.error(f"Unknown transport type: {config.transport}")
                continue
            mcp_client = MCPClient(server_params=server_params, mcp_name=name)
            self.regist_clients[name] = mcp_client
            logging.info(f"Registered mcp client: {name}")

    async def register_tools(self, llm: LLMProcessor) -> ToolsSchema:
        all_standard_tools = []
        for _, client in self.regist_clients.items():
            tools_schema = await client.register_tools(llm)
            tools_schema.standard_tools and all_standard_tools.extend(tools_schema.standard_tools)
            # todo: custom tools e.g.:{"adapter_type": [{"tool_name": "tool_description"}, ...], ...}"}
        return ToolsSchema(standard_tools=all_standard_tools)


class MCPClient(EventHandlerManager):
    def __init__(
        self,
        server_params: Union[StdioServerParameters, SseServerParameters, StreamableHttpParameters],
        **kwargs,
    ):
        super().__init__()
        self._mcp_name = kwargs.get("mcp_name", "mcp")
        self._server_params = server_params
        self._session = ClientSession
        if isinstance(server_params, StdioServerParameters):
            self._client = stdio_client
        elif isinstance(server_params, SseServerParameters):
            self._client = sse_client
            self._register_tools = self._sse_register_tools
        elif isinstance(server_params, StreamableHttpParameters):
            self._client = streamablehttp_client
            self._register_tools = self._streamable_http_register_tools
        else:
            raise TypeError(
                f"{self} invalid argument type: `server_params` must be either StdioServerParameters or an SSE server url string."
            )

    async def register_tools(self, llm: LLMProcessor) -> ToolsSchema:
        tools_schema = await self._register_tools(llm)
        return tools_schema

    def _convert_mcp_schema(self, tool_name: str, tool_schema: Dict[str, Any]) -> FunctionSchema:
        """Convert an mcp tool schema to FunctionSchema format.
        Args:
            tool_name: The name of the tool
            tool_schema: The mcp tool schema
        Returns:
            A FunctionSchema instance
        """

        logging.debug(f"Converting schema for tool '{tool_name}'")
        logging.trace(f"Original schema: {json.dumps(tool_schema, indent=2)}")

        properties = tool_schema["input_schema"].get("properties", {})
        required = tool_schema["input_schema"].get("required", [])

        schema = FunctionSchema(
            name=tool_name,
            description=tool_schema["description"],
            properties=properties,
            required=required,
        )

        logging.trace(f"Converted schema: {json.dumps(schema.to_default_dict(), indent=2)}")

        return schema

    async def _streamable_http_register_tools(self, llm: LLMProcessor) -> ToolsSchema:
        """Register all available streamable http mcp server tools with the LLM processor.
        Args:
            llm: The achatbot LLM processor to register tools with
        Returns:
            A ToolsSchema containing all registered tools
        """

        async def mcp_tool_wrapper(
            function_name: str,
            tool_call_id: str,
            arguments: Dict[str, Any],
            llm: LLMProcessor,
            context: any,
            result_callback: any,
        ) -> None:
            logging.debug(f"Executing tool '{function_name}' with call ID: {tool_call_id}")
            logging.trace(f"Tool arguments: {json.dumps(arguments, indent=2)}")
            try:
                async with self._client(
                    url=self._server_params.url,
                    headers=self._server_params.headers,
                    timeout=self._server_params.timeout,
                    sse_read_timeout=self._server_params.sse_read_timeout,
                    terminate_on_close=self._server_params.terminate_on_close,
                    auth=None,  # todo if use auth mcp server(or third party auth server)
                ) as (read, write, _):
                    async with self._session(read, write) as session:
                        await session.initialize()
                        await self._call_tool(session, function_name, arguments, result_callback)
            except Exception as e:
                error_msg = f"Error calling mcp {self._mcp_name} tool {function_name}: {str(e)}"
                logging.error(error_msg)
                logging.exception("Full exception details:")
                await result_callback(error_msg)

        async with self._client(
            url=self._server_params.url,
            headers=self._server_params.headers,
            timeout=self._server_params.timeout,
            sse_read_timeout=self._server_params.sse_read_timeout,
            terminate_on_close=self._server_params.terminate_on_close,
            auth=None,  # todo if use auth mcp server(or third party auth server)
        ) as (read, write, _):
            async with self._session(read, write) as session:
                await session.initialize()
                tools_schema = await self._list_tools(session, mcp_tool_wrapper, llm)
                return tools_schema

    async def _sse_register_tools(self, llm: LLMProcessor) -> ToolsSchema:
        """Register all available mcp sse server tools with the LLM processor.
        Args:
            llm: The achatbot LLM processor to register tools with
        Returns:
            A ToolsSchema containing all registered tools
        """

        async def mcp_tool_wrapper(
            function_name: str,
            tool_call_id: str,
            arguments: Dict[str, Any],
            llm: LLMProcessor,
            context: any,
            result_callback: any,
        ) -> None:
            logging.debug(f"Executing tool '{function_name}' with call ID: {tool_call_id}")
            logging.trace(f"Tool arguments: {json.dumps(arguments, indent=2)}")
            try:
                async with self._client(
                    url=self._server_params.url,
                    headers=self._server_params.headers,
                    timeout=self._server_params.timeout,
                    sse_read_timeout=self._server_params.sse_read_timeout,
                    auth=None,  # todo if use auth mcp server(or third party auth server)
                ) as (read, write):
                    async with self._session(read, write) as session:
                        await session.initialize()
                        await self._call_tool(session, function_name, arguments, result_callback)
            except Exception as e:
                error_msg = f"Error calling mcp {self._mcp_name} tool {function_name}: {str(e)}"
                logging.error(error_msg)
                logging.exception("Full exception details:")
                await result_callback(error_msg)

        async with self._client(
            url=self._server_params.url,
            headers=self._server_params.headers,
            timeout=self._server_params.timeout,
            sse_read_timeout=self._server_params.sse_read_timeout,
            auth=None,  # todo if use auth mcp server(or third party auth server)
        ) as (read, write):
            async with self._session(read, write) as session:
                await session.initialize()
                tools_schema = await self._list_tools(session, mcp_tool_wrapper, llm)
                return tools_schema

    async def _register_tools(self, llm: LLMProcessor) -> ToolsSchema:
        """Register all available mcp streamable http server tools with the LLM processor.
        Args:
            llm: The achatbot LLM processor to register tools with
        Returns:
            A ToolsSchema containing all registered tools
        """

        async def mcp_tool_wrapper(
            function_name: str,
            tool_call_id: str,
            arguments: Dict[str, Any],
            llm: LLMProcessor,
            context: any,
            result_callback: any,
        ) -> None:
            logging.debug(f"Executing tool '{function_name}' with call ID: {tool_call_id}")
            logging.trace(f"Tool arguments: {json.dumps(arguments, indent=2)}")
            try:
                async with self._client(self._server_params) as (read, write):
                    async with self._session(read, write) as session:
                        await session.initialize()
                        await self._call_tool(session, function_name, arguments, result_callback)
            except Exception as e:
                error_msg = f"Error calling mcp {self._mcp_name} tool {function_name}: {str(e)}"
                logging.error(error_msg)
                logging.exception("Full exception details:")
                await result_callback(error_msg)

        async with self._client(self._server_params) as (read, write):
            async with self._session(read, write) as session:
                await session.initialize()
                tools_schema = await self._list_tools(session, mcp_tool_wrapper, llm)
                return tools_schema

    async def _call_tool(self, session, function_name, arguments, result_callback):
        logging.debug(f"Calling mcp tool '{function_name}'")
        try:
            results = await session.call_tool(function_name, arguments=arguments)
        except Exception as e:
            error_msg = f"Error calling mcp tool {function_name}: {str(e)}"
            logging.error(error_msg)

        response = ""
        if results:
            if hasattr(results, "content") and results.content:
                for i, content in enumerate(results.content):
                    if hasattr(content, "text") and content.text:
                        logging.debug(f"Tool response chunk {i}: {content.text}")
                        response += content.text
                    else:
                        # logging.debug(f"Non-text result content: '{content}'")
                        pass
                logging.info(f"MCP {self._mcp_name} Tool '{function_name}' completed successfully")
                logging.debug(f"Final response: {response}")
            else:
                logging.error(
                    f"MCP {self._mcp_name}Error getting content from {function_name} results."
                )

        final_response = response if len(response) else "Sorry, could not call the mcp tool"
        await result_callback(final_response)

    async def _list_tools(self, session: ClientSession, mcp_tool_wrapper, llm: LLMProcessor):
        available_tools = await session.list_tools()
        if available_tools is None or available_tools.tools is None:
            logging.warning(f"MCP {self._mcp_name} No tools found")
            return ToolsSchema(standard_tools=[])

        tool_schemas: List[FunctionSchema] = []
        for tool in available_tools.tools:
            tool_name = tool.name
            try:
                # Convert the schema
                function_schema = self._convert_mcp_schema(
                    tool_name,
                    {"description": tool.description, "input_schema": tool.inputSchema},
                )

                if llm.has_function(tool_name):
                    logging.warning(
                        f"MCP {self._mcp_name} {tool_name} is already registered in LLMProcessor, over registering"
                    )
                # Register the wrapped function
                llm.register_function(tool_name, mcp_tool_wrapper)

                # Add to list of schemas
                tool_schemas.append(function_schema)
                logging.info(
                    f"MCP {self._mcp_name} Successfully registered tool '{tool_name}' | Tool description: {tool.description} | Tool schema: {tool.inputSchema}"
                )

            except Exception as e:
                logging.error(
                    f"MCP {self._mcp_name} Failed to register tool '{tool_name}': {str(e)}"
                )
                logging.exception("Full exception details:")
                continue

        logging.info(f"MCP {self._mcp_name} Completed registration of {len(tool_schemas)} tools")
        # {
        #    logging.info(f"tool_schemas[{i}]: {str(item)}") for i, item in enumerate(tool_schemas)
        # } if tool_schemas else logging.info(tool_schemas)
        tools_schema = ToolsSchema(standard_tools=tool_schemas)

        return tools_schema
