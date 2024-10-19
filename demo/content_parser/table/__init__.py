import os

match os.getenv("TABLE_MODEL"):
    case "podcast":
        from . import podcast as table
    case _:
        from . import chapter as table
