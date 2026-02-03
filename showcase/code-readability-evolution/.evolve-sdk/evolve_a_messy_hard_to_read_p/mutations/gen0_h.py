"""Variant H: Hybrid approach with early whale detection and dict-based dispatch."""
from typing import Any


WHALE_THRESHOLD = 10_000
TOP_CUSTOMERS_LIMIT = 5


def summarize_transactions(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce a financial summary from raw transaction records.

    Uses dict-based dispatch to handle sale/refund logic cleanly,
    with early whale detection during accumulation.

    Args:
        transactions: list of dicts, each with type, amount, customer, category.

    Returns:
        Summary dict with total, count, avg, top_customers, categories, has_whale.

    Example:
        >>> summarize_transactions([{"type": "sale", "amount": 500, "customer": "X", "category": "Y"}])
    """
    total = 0.0
    count = len(transactions)
    customer_spending: dict[str, float] = {}
    categories: dict[str, list[float, int]] = {}
    has_whale = False

    amount_sign = {"sale": 1.0, "refund": -1.0}

    for txn in transactions:
        amount = txn["amount"]
        customer = txn["customer"]
        signed_amount = amount * amount_sign[txn["type"]]
        total += signed_amount

        current_spend = customer_spending.get(customer, 0.0) + signed_amount
        customer_spending[customer] = current_spend

        if current_spend > WHALE_THRESHOLD:
            has_whale = True

        if txn["type"] == "sale":
            category = txn["category"]
            if category in categories:
                entry = categories[category]
                entry[0] += amount
                entry[1] += 1
            else:
                categories[category] = [amount, 1]

    average = total / count if count else 0.0

    ranked_customers = sorted(
        customer_spending.items(), key=lambda pair: pair[1], reverse=True
    )

    category_summary = {
        name: {"revenue": data[0], "orders": data[1], "avg": data[0] / data[1]}
        for name, data in categories.items()
    }

    return {
        "total": total,
        "count": count,
        "avg": average,
        "top_customers": ranked_customers[:TOP_CUSTOMERS_LIMIT],
        "categories": category_summary,
        "has_whale": has_whale,
    }
