"""Variant D: Dataclass-based approach with named tuples for clarity."""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


WHALE_THRESHOLD = 10_000
MAX_TOP_CUSTOMERS = 5


@dataclass
class TransactionAggregator:
    """Accumulates transaction data in a single pass."""

    total: float = 0.0
    count: int = 0
    customer_spending: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    category_revenue: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    category_orders: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def process(self, txn: dict[str, Any]) -> None:
        """Process a single transaction."""
        amount = txn["amount"]
        customer = txn["customer"]
        if txn["type"] == "sale":
            self.total += amount
            self.customer_spending[customer] += amount
            category = txn["category"]
            self.category_revenue[category] += amount
            self.category_orders[category] += 1
        else:
            self.total -= amount
            self.customer_spending[customer] -= amount
        self.count += 1

    def build_summary(self) -> dict[str, Any]:
        """Construct the final summary dictionary."""
        average = self.total / self.count if self.count else 0.0

        ranked = sorted(self.customer_spending.items(), key=lambda pair: pair[1], reverse=True)

        categories = {
            cat: {
                "revenue": self.category_revenue[cat],
                "orders": self.category_orders[cat],
                "avg": self.category_revenue[cat] / self.category_orders[cat],
            }
            for cat in self.category_revenue
        }

        has_whale = any(
            spending > WHALE_THRESHOLD for spending in self.customer_spending.values()
        )

        return {
            "total": self.total,
            "count": self.count,
            "avg": average,
            "top_customers": ranked[:MAX_TOP_CUSTOMERS],
            "categories": categories,
            "has_whale": has_whale,
        }


def summarize_transactions(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize transactions using an aggregator pattern.

    Args:
        transactions: list of dicts with 'type', 'amount', 'customer', 'category'.

    Returns:
        Summary dict with total, count, avg, top_customers, categories, has_whale.

    Example:
        >>> summarize_transactions([{"type": "sale", "amount": 100, "customer": "A", "category": "X"}])
    """
    aggregator = TransactionAggregator()
    for txn in transactions:
        aggregator.process(txn)
    return aggregator.build_summary()
