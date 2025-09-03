import asyncio


async def run():
    await asyncio.sleep(1)


def sync_run():
    loop = asyncio.get_event_loop()
    if loop.is_running():
        print("create_task to run")
        loop.run_until_complete(run())
    else:
        print("asyncio.run")
        asyncio.run(run())


async def async_main():
    sync_run()


if __name__ == "__main__":
    import nest_asyncio

    nest_asyncio.apply()
    asyncio.run(async_main())
    sync_run()
