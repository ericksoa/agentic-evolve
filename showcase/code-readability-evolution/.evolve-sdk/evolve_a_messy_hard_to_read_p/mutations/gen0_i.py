"""Variant I: Reduce-style accumulator with operator.itemgetter for key access."""
import operator
from typing import Any


WHALE_THRESHOLD = 10_000
MAX_TOP_CUSTOMERS = 5


def summarize_transactions(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a transaction summary using efficient key accessors.

    Uses operator.itemgetter for fast dict key access and a single accumulation pass.

    Args:
        transactions: list of dicts with type, amount, customer, category keys.

    Returns:
        Summary with total, count, avg, top_customers, categories, has_whale.

    Example:
        >>> summarize_transactions([{"type": "sale", "amount": 42, "customer": "C", "category": "D"}])
    """
    get_fields = operator.itemgetter("type", "amount", "customer", "category")

    total = 0.0
    count = len(transactions)
    customer_spending: dict[str, float] = {}
    category_revenue: dict[str, float] = {}
    category_orders: dict[str, int] = {}
    has_whale = False

    for txn in transactions:
        txn_type, amount, customer, category = get_fields(txn)

        if txn_type == "sale":
            total += amount
            new_spend = customer_spending.get(customer, 0.0) + amount
            customer_spending[customer] = new_spend
            if new_spend > WHALE_THRESHOLD:
                has_whale = True
            category_revenue[category] = category_revenue.get(category, 0.0) + amount
            category_orders[category] = category_orders.get(category, 0) + 1
        else:
            total -= amount
            customer_spending[customer] = customer_spending.get(customer, 0.0) - amount

    average = total / count if count else 0.0

    top_customers = sorted(
        customer_spending.items(), key=operator.itemgetter(1), reverse=True
    )[:MAX_TOP_CUSTOMERS]

    categories = {
        name: {
            "revenue": category_revenue[name],
            "orders": category_orders[name],
            "avg": category_revenue[name] / category_orders[name],
        }
        for name in category_revenue
    }

    return {
        "total": total,
        "count": count,
        "avg": average,
        "top_customers": top_customers,
        "categories": categories,
        "has_whale": has_whale,
    }
