import anyio

from mcp.shared.context import RequestContext
import mcp.types as types

from .tool_register import functions


@functions.register("stateless_send_notification")
async def stateless_send_notification(
    ctx: RequestContext,
    interval: float = 1.0,
    count: int = 5,
    caller: str = "unknown",
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    # Send the specified number of notifications with the given interval
    for i in range(count):
        await ctx.session.send_log_message(
            level="info",
            data=f"Notification {i+1}/{count} from caller: {caller}",
            logger="notification_stream",
            related_request_id=ctx.request_id,
        )
        if i < count - 1:  # Don't wait after the last notification
            await anyio.sleep(interval)

    return [
        types.TextContent(
            type="text",
            text=(f"Sent {count} notifications with {interval}s interval" f" for caller: {caller}"),
        )
    ]
