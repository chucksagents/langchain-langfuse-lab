import json
from pathlib import Path
from langchain_core.tools import tool


DATA_FILE = Path(__file__).parent.parent / "data" / "mock-financial-dispute-data.json"


def load_data():
    with open(DATA_FILE, "r") as f:
        return json.load(f)


# Define tools
@tool
def get_transaction(transaction_id: str) -> dict:
    """Retrieve information about a card transaction."""
    data = load_data()

    for transaction in data["transactions"]:
        if transaction["transaction_id"] == transaction_id:
            return transaction
    return {"error": f"Transaction {transaction_id} not found"}


@tool
def get_customer(customer_id: str) -> dict:
    """Retrieve profile information for a customer by their customer ID."""
    data = load_data()

    for customer in data["customers"]:
        if customer["customer_id"] == customer_id:
            return customer
    return {"error": f"Customer {customer_id} not found"}


@tool
def get_customer_transactions(customer_id: str) -> list:
    """Retrieve all transactions for a given customer ID."""
    data = load_data()

    transactions = [t for t in data["transactions"] if t["customer_id"] == customer_id]
    if not transactions:
        return [{"error": f"No transactions found for customer {customer_id}"}]
    return transactions
