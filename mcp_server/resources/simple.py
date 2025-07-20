from pydantic import AnyUrl

from .resource_register import resources, SAMPLE_RESOURCES


@resources.register("/help.txt")
@resources.register("/about.txt")
@resources.register("/greeting.txt")
async def read_resource(uri: AnyUrl) -> str | bytes:
    if uri.path is None:
        raise ValueError(f"Invalid resource path: {uri}")
    name = uri.path.replace(".txt", "").lstrip("/")

    if name not in SAMPLE_RESOURCES:
        raise ValueError(f"Unknown resource: {uri}")

    return SAMPLE_RESOURCES[name]
