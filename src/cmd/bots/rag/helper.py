import os
import logging


def get_tidb_url():
    url = f"mysql+pymysql://{os.getenv('TIDB_USERNAME')}:{os.getenv('TIDB_PASSWORD')}@{os.getenv('TIDB_HOST')}:{os.getenv('TIDB_PORT')}/{os.getenv('TIDB_DATABASE')}?ssl_ca={os.getenv('TIDB_SSL_CA')}&ssl_verify_cert=true&ssl_verify_identity=true"
    logging.info(f"tidb url: {url}")
    return url
