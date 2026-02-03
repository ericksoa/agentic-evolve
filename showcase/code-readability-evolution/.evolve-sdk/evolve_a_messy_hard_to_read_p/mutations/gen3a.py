"""Branchless single-pass aggregation with defaultdict."""
from collections import defaultdict
from typing import Any

WHALE_THRESHOLD = 10_000
SALE_TYPE = "sale"
SIGN_MAP = {SALE_TYPE: 1.0, "refund": -1.0}


def summarize_transactions(
    transactions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize transactions into aggregate metrics.

    Args:
        transactions: dicts with type, amount,
            customer, category keys.

    Returns:
        Summary with total, count, avg,
        top_customers, categories, has_whale.

    Example:
        >>> result = summarize_transactions([])
        >>> result["total"]
        0.0
    """
    total = 0.0
    count = len(transactions)
    customer_spending: dict[str, float] = defaultdict(float)
    category_revenue: dict[str, float] = defaultdict(float)
    category_orders: dict[str, int] = defaultdict(int)

    for txn in transactions:
        sign = SIGN_MAP[txn["type"]]
        signed_amount = sign * txn["amount"]
        total += signed_amount
        customer_spending[txn["customer"]] += signed_amount
        is_sale = sign > 0
        category = txn.get("category", "")
        category_revenue[category] += txn["amount"] * is_sale
        category_orders[category] += is_sale

    category_revenue.pop("", None)
    category_orders.pop("", None)

    average = total / count if count else 0.0

    sorted_customers = sorted(
        customer_spending.items(),
        key=lambda pair: pair[1],
        reverse=True,
    )
    top_customers = [
        (name, spent)
        for name, spent in sorted_customers[:5]
    ]

    categories = {
        name: {
            "revenue": category_revenue[name],
            "orders": category_orders[name],
            "avg": category_revenue[name] / category_orders[name],
        }
        for name in category_orders
        if category_orders[name] > 0
    }

    has_whale = any(
        spent > WHALE_THRESHOLD
        for spent in customer_spending.values()
    )

    return {
        "total": total,
        "count": count,
        "avg": average,
        "top_customers": top_customers,
        "categories": categories,
        "has_whale": has_whale,
    }
