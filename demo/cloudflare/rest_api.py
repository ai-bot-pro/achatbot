import os
import json
import logging
import http.client
from typing import List

import typer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

app = typer.Typer()


@app.command("d1_table_query")
def d1_table_query(db_id: str, sql: str, sql_params: List[str] = []) -> dict:
    """
    https://developers.cloudflare.com/api/operations/cloudflare-d1-query-database
    """
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_key = os.getenv("CLOUDFLARE_API_KEY")

    payload = {
        "params": sql_params,
        "sql": sql,
    }
    body = json.dumps(payload)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    conn = http.client.HTTPSConnection("api.cloudflare.com")
    conn.request(
        "POST",
        f"/client/v4/accounts/{account_id}/d1/database/{db_id}/query",
        body,
        headers,
    )
    res = conn.getresponse()
    data = res.read().decode("utf-8")
    # print(data)
    json_data = json.loads(data)
    # logging.info(f"body:{body}, db_id:{db_id}, query res:{json_data}")
    return json_data


@app.command("d1_db")
def d1_db(db_id: str) -> dict:
    """
    https://developers.cloudflare.com/api/operations/cloudflare-d1-get-database
    """
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_key = os.getenv("CLOUDFLARE_API_KEY")

    conn = http.client.HTTPSConnection("api.cloudflare.com")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    conn.request(
        "GET",
        f"/client/v4/accounts/{account_id}/d1/database/{db_id}",
        headers=headers,
    )

    res = conn.getresponse()
    data = res.read()

    data = res.read().decode("utf-8")
    json_data = json.loads(data)
    logging.info(f"get db_id:{db_id}, query res:{json_data}")
    return json_data


r"""
python -m demo.cloudflare.rest_api d1_db \
    09f7a7c7-66ae-41ea-9dbe-b8b635b19758

python -m demo.cloudflare.rest_api d1_table_query \
    09f7a7c7-66ae-41ea-9dbe-b8b635b19758 \
    "select * from podcast limit 1"
"""
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    app()
