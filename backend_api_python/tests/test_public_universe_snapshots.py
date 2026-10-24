import pandas as pd

from scripts import refresh_public_universe_snapshots as snapshots


class _RowsCursor:
    def __init__(self, rows):
        self.rows = rows
        self.params = None
        self.executions = []

    def execute(self, sql, params=None):
        self.params = params
        self.executions.append((sql, params))

    def fetchall(self):
        return self.rows

    def close(self):
        pass


class _RowsConnection:
    def __init__(self, rows):
        self.cursor_obj = _RowsCursor(rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        pass


def test_hsi_api_parser_uses_requested_parent_index_and_normalizes_codes():
    payload = {
        "data": {
            "viewConsDetail": {
                "00001.00": [
                    {
                        "stockCode": "700",
                        "stockName": "TENCENT",
                        "exchangeListed": "Hong Kong Stock Exchange",
                        "stockType": "HK",
                        "dateTime": "2026-10-23 16:08:30",
                    },
                    {"stockCode": "939", "stockName": "CCB"},
                ],
                "00001.01": [{"stockCode": "5", "stockName": "HSBC"}],
            },
        },
    }

    rows = snapshots._parse_hsi_constituent_payload(payload, "00001.00", "hsi")

    assert [row["symbol"] for row in rows] == ["00700", "00939"]
    assert rows[0]["metadata"]["series_code"] == "hsi"
    assert rows[0]["metadata"]["source_timestamp"] == "2026-10-23 16:08:30"


def test_hk_stock_connect_parser_normalizes_codes():
    rows = snapshots._parse_hk_stock_connect_items([
        {"f12": "700", "f14": "Tencent"},
        {"f12": "09988", "f14": "Alibaba"},
        {"f12": "invalid", "f14": "Ignored"},
    ])

    assert [row["symbol"] for row in rows] == ["00700", "09988"]
    assert rows[0]["metadata"]["source"] == "eastmoney_hk_stock_connect"


def test_json_decoder_accepts_gb18030_market_names():
    payload = '{"data":{"diff":[{"f12":"00700","f14":"腾讯控股"}]}}'.encode("gb18030")

    decoded = snapshots._decode_json_bytes(payload)

    assert decoded["data"]["diff"][0]["f14"] == "腾讯控股"


def test_etf_snapshot_loader_reads_hot_symbol_master_rows(monkeypatch):
    connection = _RowsConnection([
        {"market": "USStock", "symbol": "spy", "name": "SPDR S&P 500 ETF"},
        {"market": "USStock", "symbol": "qqq", "name": "Invesco QQQ"},
    ])
    monkeypatch.setattr(snapshots, "get_db_connection", lambda: connection)

    rows = snapshots.symbol_master_etfs("USStock")

    assert connection.cursor_obj.params == ("USStock",)
    repair_sql, repair_params = connection.cursor_obj.executions[0]
    assert "SET asset_class = 'etf', is_hot = 1" in repair_sql
    assert repair_params[0] == "USStock"
    assert "SPY" in repair_params[1]
    assert [row["symbol"] for row in rows] == ["SPY", "QQQ"]
    assert rows[0]["rank"] == 1
    assert rows[0]["metadata"] == {"source": "symbol_master", "asset_class": "etf"}
    assert "hk_etf" in snapshots.LOADERS
    assert "us_etf" in snapshots.LOADERS


def test_dow30_parser_selects_company_symbol_table():
    frame = pd.DataFrame({
        "Company": [f"Company {index}" for index in range(30)],
        "Symbol": [f"D{index:02d}" for index in range(30)],
        "Industry": ["Industry"] * 30,
        "Exchange": ["NYSE"] * 30,
    })

    rows = snapshots._parse_dow30_tables([pd.DataFrame({"Other": [1]}), frame])

    assert len(rows) == 30
    assert rows[0]["symbol"] == "D00"
    assert rows[0]["metadata"] == {"industry": "Industry", "exchange": "NYSE"}


def test_us_index_parser_selects_expected_table_and_normalizes_class_symbols():
    frame = pd.DataFrame({
        "Company": [f"Company {index}" for index in range(400)],
        "Symbol": ["BRK.B", *[f"M{index:03d}" for index in range(1, 400)]],
        "GICS Sector": ["Financials"] * 400,
        "GICS Sub-Industry": ["Multi-Sector Holdings"] * 400,
    })

    rows = snapshots._parse_us_index_tables([pd.DataFrame({"Other": [1]}), frame], (395, 405))

    assert len(rows) == 400
    assert rows[0]["symbol"] == "BRK-B"
    assert rows[0]["metadata"]["sector"] == "Financials"


def test_common_universe_loaders_are_registered():
    assert snapshots.EXPECTED_MEMBER_COUNTS["sse50"] == (50, 50)
    assert snapshots.EXPECTED_MEMBER_COUNTS["csi800"] == (800, 800)
    assert snapshots.EXPECTED_MEMBER_COUNTS["csi1000"] == (1000, 1000)
    assert snapshots.EXPECTED_MEMBER_COUNTS["dow30"] == (30, 30)
    assert snapshots.EXPECTED_MEMBER_COUNTS["sp400"] == (395, 405)
    assert snapshots.EXPECTED_MEMBER_COUNTS["sp600"] == (595, 610)
    assert snapshots.EXPECTED_MEMBER_COUNTS["russell1000"] == (950, 1050)
    assert {"sse50", "csi800", "csi1000", "dow30", "sp400", "sp600", "russell1000"}.issubset(
        snapshots.LOADERS
    )
    assert {"hk_hsi", "hk_tech30", "hk_china_enterprises50", "hk_stock_connect"}.issubset(
        snapshots.LOADERS
    )
    assert set(snapshots.US_SECTOR_UNIVERSES).issubset(snapshots.LOADERS)


def test_sector_loader_reuses_sp500_download_and_preserves_sector_metadata(monkeypatch):
    calls = {"count": 0}

    def fake_sp500():
        calls["count"] += 1
        return [
            {"market": "USStock", "symbol": "A", "rank": 1, "metadata": {"sector": "Energy"}},
            {"market": "USStock", "symbol": "B", "rank": 2, "metadata": {"sector": "Financials"}},
            {"market": "USStock", "symbol": "C", "rank": 3, "metadata": {"sector": "Energy"}},
        ]

    monkeypatch.setitem(snapshots.LOADERS, "sp500", fake_sp500)
    cache = {}

    energy = snapshots.load_universe_members("us_sector_energy", cache)
    financials = snapshots.load_universe_members("us_sector_financials", cache)

    assert calls["count"] == 1
    assert [row["symbol"] for row in energy] == ["A", "C"]
    assert [row["rank"] for row in energy] == [1, 2]
    assert [row["symbol"] for row in financials] == ["B"]
