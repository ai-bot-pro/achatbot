from .. import app, types
from pydantic import AnyUrl

SAMPLE_RESOURCES = {
    "greeting": "Hello! This is a sample text resource.",
    "help": "This server provides a few sample text resources for testing.",
    "about": "This is the simple-resource MCP server implementation.",
}


@app.list_resources()
async def list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri=AnyUrl(f"file:///{name}.txt"),
            name=name,
            description=f"A sample text resource named {name}",
            mimeType="text/plain",
        )
        for name in SAMPLE_RESOURCES.keys()
    ]


@app.read_resource()
async def read_resource(uri: AnyUrl) -> str | bytes:
    if uri.path is None:
        raise ValueError(f"Invalid resource path: {uri}")
    name = uri.path.replace(".txt", "").lstrip("/")

    if name not in SAMPLE_RESOURCES:
        raise ValueError(f"Unknown resource: {uri}")

    return SAMPLE_RESOURCES[name]

@app.subscribe_resource()
async def update_resource(uri: AnyUrl) -> str | bytes:
    if uri.path is None:
        raise ValueError(f"Invalid resource path: {uri}")
    name = uri.path.replace(".txt", "").lstrip("/")

    if name not in SAMPLE_RESOURCES:
        raise ValueError(f"Unknown resource: {uri}")

    return SAMPLE_RESOURCES[name]