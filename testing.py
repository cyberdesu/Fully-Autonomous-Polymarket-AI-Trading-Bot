from py_clob_client.client import ClobClient
import os

host = "https://clob.polymarket.com"
chain_id = 137  # Polygon mainnet
private_key = "0x702ccdb0eee7330ba06c616c27c2a986ce0230844d458f26c8a8e43ed3ea2870"

# Derive API credentials
temp_client = ClobClient(host, key=private_key, chain_id=chain_id)
api_creds = temp_client.create_or_derive_api_creds()

# Initialize trading client
client = ClobClient(
    host,
    key=private_key,
    chain_id=chain_id,
    creds=api_creds,
    signature_type=1,  # EOA
    funder="0x34a7Ef33A42B2C36527A564470A7a54E4662b81f",  # same as private key address
)

from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY

response = client.create_and_post_order(
    OrderArgs(
        token_id="YOUR_TOKEN_ID",
        price=0.50,
        size=10,
        side=BUY,
    ),
    options={
        "tick_size": "0.01",
        "neg_risk": False,  # Set to True for multi-outcome markets
    },
    order_type=OrderType.GTC
)

print("Order ID:", response["orderID"])
print("Status:", response["status"])