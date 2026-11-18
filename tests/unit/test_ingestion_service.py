from fincomb.services.ingestion_service import group_by_source, normalise_symbols


def test_normalise_symbols_keeps_order_and_dedupes():
    assert normalise_symbols([" aapl", "SPY", "AAPL", "", "  "]) == ["AAPL", "SPY"]


def test_group_by_source_owner_then_default():
    owners = {"AAPL": "tiingo", "SPY": None}
    groups = group_by_source(["aapl", "spy", "msft"], None, owners, "yfinance")
    assert groups == {"tiingo": ["AAPL"], "yfinance": ["SPY", "MSFT"]}


def test_group_by_source_explicit_source_wins():
    groups = group_by_source(["AAPL", "SPY"], "csv", {"AAPL": "tiingo"}, "yfinance")
    assert groups == {"csv": ["AAPL", "SPY"]}
