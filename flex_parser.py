"""
Parse IBKR Flex Query Activity Statement CSV.
Multi-section format: each section starts with a header row whose first field = 'ClientAccountID'.
Returns a dict of section_name → pd.DataFrame.
"""
import csv
import math
import pandas as pd
from io import StringIO

# ── Section identification by distinctive column signature ─────────────────────
_SECTION_SIGNATURES = {
    "nav":                {"ReportDate", "Cash", "Stock", "Total"},
    "trades":             {"TradeID", "Buy/Sell", "FifoPnlRealized", "TradePrice"},
    "positions":          {"Quantity", "MarkPrice", "FifoPnlUnrealized", "CostBasisMoney"},
    "realized_pnl":       {"RealizedShortTermProfit", "TotalRealizedPnl"},
    "mtm_pnl":            {"PreviousCloseQuantity", "TransactionMtmPnl"},
    "period_pnl":         {"Mark-to-Market MTD", "RealizedPnlMTD"},
    "dividends":          {"DividendType", "Amount"},
    "fx_activity":        {"ActivityDescription", "FXCurrency", "RealizedP/L"},
    "interest":           {"StartingAccrualBalance", "InterestAccrued"},
    "pnl_summary":        {"StartingValue", "Mtm", "Realized", "ChangeInUnrealized"},
    "cash_summary":       {"StartingCash", "ClientFees"},
    # Prior Period Positions — daily Date+PriorMtmPnl rows per symbol
    "prior_period_pnl":   {"PriorMtmPnl", "Date", "Price"},
    # Asset-class MTM bridge (Prior Period Value / EndOfPeriodValue)
    "mtm_asset_class":    {"MtmPnlPriorPeriodPositions", "Prior Period Value"},
    # Account info — single row, not used for P&L
    "account_info":       {"DateOpened", "IBEntity", "TaxLotMatchingMethod"},
    # FX open position lots — not used for P&L
    "fx_positions":       {"LotDescription", "LotOpenDateTime", "FXCurrency"},
    # Statement of Funds — daily per-currency cash ledger with running Balance
    "statement_of_funds": {"Balance", "ActivityDescription"},
}


def _identify_section(cols: list) -> str | None:
    col_set = set(cols)
    for name, sig in _SECTION_SIGNATURES.items():
        if sig.issubset(col_set):
            return name
    return None


def parse_flex_csv(filepath_or_buffer) -> dict:
    """Parse IBKR Flex multi-section CSV → dict of section_name → DataFrame."""
    if hasattr(filepath_or_buffer, "read"):
        content = filepath_or_buffer.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8-sig")
    else:
        with open(filepath_or_buffer, "r", encoding="utf-8-sig") as f:
            content = f.read()

    reader = csv.reader(StringIO(content))

    sections: dict[str, list] = {}    # section_name → list of row dicts
    current_cols: list | None  = None
    current_section: str | None = None
    _raw_idx = 0  # counter for unrecognised sections

    for row in reader:
        if not row:
            continue
        first = row[0].strip()

        if first == "ClientAccountID":
            # New section header — identify it; fall back to _raw_N so nothing is dropped
            current_cols = row
            name = _identify_section(row)
            if name is None:
                _raw_idx += 1
                name = f"_raw_{_raw_idx}"
            current_section = name

        elif first in ("", "-"):
            pass  # subtotal / empty rows — skip

        else:
            # Data row
            if current_cols and current_section:
                padded = row[:len(current_cols)] + [""] * max(0, len(current_cols) - len(row))
                row_dict = dict(zip(current_cols, padded))
                sections.setdefault(current_section, []).append(row_dict)

    # Convert each section to DataFrame and clean
    result = {name: pd.DataFrame(rows) for name, rows in sections.items()}
    _clean_sections(result)
    return result


# ── Per-section type coercion ──────────────────────────────────────────────────

def _to_num(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _clean_sections(sections: dict):
    if "nav" in sections:
        df = sections["nav"]
        df["ReportDate"] = pd.to_datetime(df["ReportDate"], errors="coerce")
        df = _to_num(df, ["Cash", "CashLong", "CashShort", "Stock", "Options",
                          "Bonds", "Commodities", "Total", "TotalLong", "TotalShort"])
        # Accumulated pulls can leave several rows per ReportDate — keep the latest
        # (dupes otherwise double the cumulative-P&L chart which reindexes onto nav dates).
        sections["nav"] = (df.dropna(subset=["ReportDate"])
                             .sort_values("ReportDate")
                             .drop_duplicates(subset=["ReportDate"], keep="last")
                             .reset_index(drop=True))

    if "trades" in sections:
        df = sections["trades"]
        df["TradeDate"] = pd.to_datetime(df["TradeDate"], errors="coerce")
        df["DateTime"]  = pd.to_datetime(df["DateTime"],  errors="coerce")
        df = _to_num(df, ["Quantity", "TradePrice", "TradeMoney", "Proceeds",
                          "IBCommission", "NetCash", "FifoPnlRealized", "MtmPnl",
                          "FXRateToBase", "CostBasis"])
        # Keep only EXECUTION-level rows
        if "LevelOfDetail" in df.columns:
            df = df[df["LevelOfDetail"] == "EXECUTION"].copy()
        sections["trades"] = df.dropna(subset=["TradeDate"]).sort_values("TradeDate").reset_index(drop=True)

    if "positions" in sections:
        df = sections["positions"]
        df["ReportDate"] = pd.to_datetime(df.get("ReportDate", pd.Series(dtype=str)),
                                          errors="coerce")
        df = _to_num(df, ["Quantity", "MarkPrice", "PositionValue", "OpenPrice",
                          "CostBasisPrice", "CostBasisMoney", "FifoPnlUnrealized",
                          "PercentOfNAV", "FXRateToBase"])
        # Keep LOT-level rows where present
        if "LevelOfDetail" in df.columns:
            lot_df = df[df["LevelOfDetail"] == "LOT"]
            sections["positions"] = lot_df.reset_index(drop=True) if not lot_df.empty else df
        else:
            sections["positions"] = df

    if "dividends" in sections:
        df = sections["dividends"]
        dt_col = "Date/Time" if "Date/Time" in df.columns else "DateTime"
        if dt_col in df.columns:
            df["Date/Time"] = pd.to_datetime(df[dt_col], errors="coerce")
        df = _to_num(df, ["Amount"])
        sections["dividends"] = df

    if "realized_pnl" in sections:
        df = sections["realized_pnl"]
        df = _to_num(df, ["TotalRealizedPnl", "TotalUnrealizedPnl", "TotalFifoPnl",
                          "RealizedShortTermProfit", "RealizedShortTermLoss",
                          "RealizedLongTermProfit", "RealizedLongTermLoss"])
        sections["realized_pnl"] = df

    if "pnl_summary" in sections:
        df = sections["pnl_summary"]
        df = _to_num(df, ["StartingValue", "Mtm", "Realized", "ChangeInUnrealized",
                          "TransferredPnlRealized", "TransferredPnlUnrealized"])
        sections["pnl_summary"] = df

    if "prior_period_pnl" in sections:
        df = sections["prior_period_pnl"]
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = _to_num(df, ["PriorMtmPnl", "Price", "FXRateToBase"])
        df["FXRateToBase"] = df["FXRateToBase"].fillna(1.0)
        sections["prior_period_pnl"] = (df.dropna(subset=["Date"])
                                          .drop_duplicates()
                                          .sort_values("Date")
                                          .reset_index(drop=True))

    if "mtm_asset_class" in sections:
        df = sections["mtm_asset_class"]
        df = _to_num(df, ["Prior Period Value", "Transactions", "MtmPnlPriorPeriodPositions",
                          "MtmPnlTransactions", "EndOfPeriodValue"])
        sections["mtm_asset_class"] = df
