"""Minimal import shim for the pinned TSPS exchange-only configuration."""

class CoinMetricsClient:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Coin Metrics is disabled for the TSPS exchange-kline run")

class DataCollection:
    pass
