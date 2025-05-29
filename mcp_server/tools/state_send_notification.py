import logging

import anyio
from mcp.types import AnyUrl
from mcp.shared.context import RequestContext
import mcp.types as types

from .tool_register import functions


@functions.register("state_send_notification")
async def state_send_notification(
    ctx: RequestContext,
    interval: float = 1.0,
    count: int = 5,
    caller: str = "unknown",
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    # Send the specified number of notifications with the given interval
    for i in range(count):
        # Include more detailed message for resumability demonstration
        notification_msg = (
            f"[{i+1}/{count}] Event from '{caller}' - "
            f"Use Last-Event-ID to resume if disconnected"
        )
        await ctx.session.send_log_message(
            level="info",
            data=notification_msg,
            logger="notification_stream",
            # Associates this notification with the original request
            # Ensures notifications are sent to the correct response stream
            # Without this, notifications will either go to:
            # - a standalone SSE stream (if GET request is supported)
            # - nowhere (if GET request isn't supported)
            related_request_id=ctx.request_id,
        )
        logging.debug(f"Sent notification {i+1}/{count} for caller: {caller}")
        if i < count - 1:  # Don't wait after the last notification
            await anyio.sleep(interval)

    # This will send a resource notificaiton though standalone SSE
    # established by GET request
    await ctx.session.send_resource_updated(uri=AnyUrl("http:///test_resource"))
    return [
        types.TextContent(
            type="text",
            text=(f"Sent {count} notifications with {interval}s interval for caller: {caller}"),
        )
    ]
