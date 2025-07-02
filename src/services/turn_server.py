import os
import aiohttp


async def get_cloudflare_turn_servers(ttl=86400):
    auth_token = os.environ.get("CLOUDFLARE_TURN_API_TOKEN")
    key_id = os.environ.get("CLOUDFLARE_TURN_TOKEN")
    url = f"https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}/credentials/generate-ice-servers"

    headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}

    data = {"ttl": ttl}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status not in [200, 201]:
                error_text = await response.text()
                raise Exception(f"error status {response.status} {error_text}")

            data = await response.json()
            return data["iceServers"]


async def get_metered_turn_servers():
    turn_name = os.environ.get("METERED_TURN_USERNAME")
    api_key = os.environ.get("METERED_TURN_API_KEY")
    url = f"https://{turn_name}/api/v1/turn/credentials?apiKey={api_key}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status not in [200, 201]:
                error_text = await response.text()
                raise Exception(f"error status {response.status} {error_text}")

            data = await response.json()
            return data
