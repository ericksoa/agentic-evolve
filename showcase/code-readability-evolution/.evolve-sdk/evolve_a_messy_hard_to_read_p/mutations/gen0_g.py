"""Variant G: Ultra-minimal with comprehensions and max readability score focus."""
from collections import defaultdict
from typing import Any


WHALE_SPENDING_THRESHOLD = 10_000
MAX_TOP_CUSTOMERS = 5


def summarize_transactions(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a list of financial transactions into key business metrics.

    Computes total revenue, transaction count, average amount, top customers
    by spending, per-category breakdowns, and flags high-value customers.

    Args:
        transactions: each transaction dict contains 'type' (sale/refund),
            'amount' (float), 'customer' (str), and 'category' (str).

    Returns:
        A summary dict containing:
            - total: net revenue (sales minus refunds)
            - count: number of transactions
            - avg: average transaction amount
            - top_customers: top 5 spenders as (name, amount) tuples
            - categories: per-category revenue, order count, and average
            - has_whale: True if any customer spent over the threshold

    Example:
        >>> summarize_transactions([{"type": "sale", "amount": 100.0, "customer": "Alice", "category": "Books"}])
    """
    transaction_count = len(transactions)
    customer_spending: dict[str, float] = defaultdict(float)
    category_revenue: dict[str, float] = defaultdict(float)
    category_order_count: dict[str, int] = defaultdict(int)

    net_total = sum(
        txn["amount"] if txn["type"] == "sale" else -txn["amount"]
        for txn in transactions
    )

    for txn in transactions:
        amount = txn["amount"]
        customer_name = txn["customer"]
        is_sale = txn["type"] == "sale"

        customer_spending[customer_name] += amount if is_sale else -amount

        if is_sale:
            category_name = txn["category"]
            category_revenue[category_name] += amount
            category_order_count[category_name] += 1

    average_amount = net_total / transaction_count if transaction_count else 0.0

    top_customers = sorted(
        customer_spending.items(), key=lambda pair: pair[1], reverse=True
    )[:MAX_TOP_CUSTOMERS]

    category_summaries = {
        name: {
            "revenue": category_revenue[name],
            "orders": category_order_count[name],
            "avg": category_revenue[name] / category_order_count[name],
        }
        for name in category_revenue
    }

    has_whale = any(
        spending > WHALE_SPENDING_THRESHOLD
        for spending in customer_spending.values()
    )

    return {
        "total": net_total,
        "count": transaction_count,
        "avg": average_amount,
        "top_customers": top_customers,
        "categories": category_summaries,
        "has_whale": has_whale,
    }
