from pydantic import BaseModel
import polars as pl
from config import DATA_TEMP


class FinancialMetrics(BaseModel):
    """Financial metrics for a company at a specific point in time."""

    return_on_equity: float | None = None
    debt_to_equity: float | None = None
    operating_margin: float | None = None
    current_ratio: float | None = None
    return_on_invested_capital: float | None = None
    asset_turnover: float | None = None


class FinancialLineItem(BaseModel):
    """Financial line items for a company at a specific point in time."""

    capital_expenditure: float | None = None
    depreciation_and_amortization: float | None = None
    net_income: float | None = None
    outstanding_shares: float | None = None
    total_assets: float | None = None
    total_liabilities: float | None = None
    shareholders_equity: float | None = None
    dividends_and_other_cash_distributions: float | None = None
    issuance_or_purchase_of_equity_shares: float | None = None
    gross_profit: float | None = None
    revenue: float | None = None
    free_cash_flow: float | None = None
    current_assets: float | None = None
    current_liabilities: float | None = None


def get_financial_metrics(
    ticker: str, year: int, quarter: int, period: str = "ttm", limit: int = 10
) -> list[FinancialMetrics]:
    """
    Get financial metrics for a company for a specific year and quarter.

    Args:
        ticker (str): Stock ticker symbol
        year (int): Year of analysis
        quarter (int): Quarter of analysis (1-4)
        period (str): Period type (ttm for trailing twelve months)
        limit (int): Number of periods to return

    Returns:
        list[FinancialMetrics]: List of financial metrics for each period, with most recent quarter first.
        metrics[0] contains the most recent quarter's metrics,
        metrics[1] contains the previous quarter's metrics, and so on.
    """
    # Get all quarters up to the specified year and quarter
    company_data = (
        sp500_data.filter(
            (pl.col("ticker") == ticker)
            & (
                (pl.col("year") < year)
                | ((pl.col("year") == year) & (pl.col("quarter") <= quarter))
            )
        )
        .sort("datadate", descending=True)  # Sort by date in descending order
        .head(limit)
    )

    if company_data.is_empty():
        return []

    metrics_list = []
    for row in company_data.iter_rows(named=True):
        # Calculate ROE
        roe = None
        if (
            row["net_income"]
            and row["shareholders_equity"]
            and row["shareholders_equity"] != 0
        ):
            roe = row["net_income"] / row["shareholders_equity"]

        # Calculate Debt to Equity
        debt_to_equity = None
        if (
            row["total_liabilities"]
            and row["shareholders_equity"]
            and row["shareholders_equity"] != 0
        ):
            debt_to_equity = row["total_liabilities"] / row["shareholders_equity"]

        # Calculate Operating Margin
        operating_margin = None
        if row["revenue"] and row["revenue"] != 0:
            operating_margin = (row["revenue"] - row.get("cost_of_goods", 0)) / row[
                "revenue"
            ]

        # Calculate Current Ratio
        current_ratio = None
        if (
            row.get("current_assets")
            and row.get("current_liabilities")
            and row["current_liabilities"] != 0
        ):
            current_ratio = row["current_assets"] / row["current_liabilities"]

        # Calculate ROIC
        roic = None
        if row["net_income"] and row["total_assets"] and row["total_assets"] != 0:
            roic = row["net_income"] / row["total_assets"]

        # Calculate Asset Turnover
        asset_turnover = None
        if row["revenue"] and row["total_assets"] and row["total_assets"] != 0:
            asset_turnover = row["revenue"] / row["total_assets"]

        metrics = FinancialMetrics(
            return_on_equity=roe,
            debt_to_equity=debt_to_equity,
            operating_margin=operating_margin,
            current_ratio=current_ratio,
            return_on_invested_capital=roic,
            asset_turnover=asset_turnover,
        )
        metrics_list.append(metrics)

    return metrics_list


def search_line_items(
    ticker: str,
    items: list[str],
    year: int,
    quarter: int,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialLineItem]:
    """
    Get specific financial line items for a company for a specific year and quarter.

    Args:
        ticker (str): Stock ticker symbol
        items (list[str]): List of financial items to retrieve
        year (int): Year of analysis
        quarter (int): Quarter of analysis (1-4)
        period (str): Period type (ttm for trailing twelve months)
        limit (int): Number of periods to return

    Returns:
        list[FinancialLineItem]: List of financial line items for each period, with most recent quarter first.
        items[0] contains the most recent quarter's line items,
        items[1] contains the previous quarter's line items, and so on.
    """
    # Get all quarters for the ticker, sorted by date
    company_data = (
        sp500_data.filter(
            (pl.col("ticker") == ticker)
            & (
                (pl.col("year") < year)
                | ((pl.col("year") == year) & (pl.col("quarter") <= quarter))
            )
        )
        .sort("datadate", descending=True)  # Sort by date in descending order
        .head(limit)
    )
    if company_data.is_empty():
        return []

    line_items_list = []
    for row in company_data.iter_rows(named=True):
        line_item = FinancialLineItem(
            capital_expenditure=row.get("capital_expenditure"),
            depreciation_and_amortization=row.get("depreciation_and_amortization"),
            net_income=row.get("net_income"),
            outstanding_shares=row.get("outstanding_shares"),
            total_assets=row.get("total_assets"),
            total_liabilities=row.get("total_liabilities"),
            shareholders_equity=row.get("shareholders_equity"),
            dividends_and_other_cash_distributions=row.get(
                "dividends_and_other_cash_distributions"
            ),
            issuance_or_purchase_of_equity_shares=row.get(
                "issuance_or_purchase_of_equity_shares"
            ),
            gross_profit=row.get("gross_profit"),
            revenue=row.get("revenue"),
            free_cash_flow=row.get("free_cash_flow"),
            current_assets=row.get("current_assets"),
            current_liabilities=row.get("current_liabilities"),
        )
        line_items_list.append(line_item)

    return line_items_list


def get_market_cap(ticker: str, year: int, quarter: int) -> float | None:
    """
    Get market capitalization for a company for a specific year and quarter.

    Args:
        ticker (str): Stock ticker symbol
        year (int): Year of analysis
        quarter (int): Quarter of analysis (1-4)

    Returns:
        float | None: Market capitalization if available, None otherwise
    """
    # Filter data for the specific ticker, year and quarter
    company_data = (
        sp500_data.filter(
            (pl.col("ticker") == ticker)
            & (pl.col("year") == year)
            & (pl.col("quarter") == quarter)
        )
        .sort("datadate", descending=True)
        .head(1)
    )

    if company_data.is_empty():
        return None

    # Get the latest row
    row = company_data.row(0, named=True)

    # Calculate market cap if we have the required data
    if row.get("outstanding_shares") and row.get("price"):
        return row["outstanding_shares"] * row["price"]

    return None


# Load the data once at module level
sp500_data = pl.read_parquet(
    DATA_TEMP / "sp500_fundamental_quarterly_quantitative_metrics.parquet.gzip"
)

# Test the functions with AAPL 2024 Q4
if __name__ == "__main__":
    ticker = "AAPL"
    year = 2024
    quarter = 4

    print(f"\nTesting functions for {ticker} {year} Q{quarter}:")
    print("-" * 50)

    # Test get_financial_metrics
    print("\n1. Financial Metrics:")
    metrics = get_financial_metrics(ticker, year, quarter)
    if metrics:
        latest = metrics[0]
        print(
            f"ROE: {latest.return_on_equity:.2%}"
            if latest.return_on_equity
            else "ROE: N/A"
        )
        print(
            f"Debt to Equity: {latest.debt_to_equity:.2f}"
            if latest.debt_to_equity
            else "Debt to Equity: N/A"
        )
        print(
            f"Operating Margin: {latest.operating_margin:.2%}"
            if latest.operating_margin
            else "Operating Margin: N/A"
        )
        print(
            f"Current Ratio: {latest.current_ratio:.2f}"
            if latest.current_ratio
            else "Current Ratio: N/A"
        )
        print(
            f"ROIC: {latest.return_on_invested_capital:.2%}"
            if latest.return_on_invested_capital
            else "ROIC: N/A"
        )
        print(
            f"Asset Turnover: {latest.asset_turnover:.2f}"
            if latest.asset_turnover
            else "Asset Turnover: N/A"
        )
    else:
        print("No financial metrics data available")

    # Test search_line_items
    print("\n2. Financial Line Items:")
    items = search_line_items(
        ticker, ["revenue", "net_income", "free_cash_flow"], year, quarter
    )
    if items:
        latest = items[0]
        print(f"Revenue: ${latest.revenue:,.0f}" if latest.revenue else "Revenue: N/A")
        print(
            f"Net Income: ${latest.net_income:,.0f}"
            if latest.net_income
            else "Net Income: N/A"
        )
        print(
            f"Free Cash Flow: ${latest.free_cash_flow:,.0f}"
            if latest.free_cash_flow
            else "Free Cash Flow: N/A"
        )
    else:
        print("No financial line items data available")

    # Test get_market_cap
    print("\n3. Market Cap:")
    market_cap = get_market_cap(ticker, year, quarter)
    print(f"Market Cap: ${market_cap:,.0f}" if market_cap else "Market Cap: N/A")


from typing import Any, Dict, Optional, Type, Callable, List, Union, TypeVar, Literal
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI  # Assuming you're using the openai-python client
import yaml
import os
import json


# Initialize client (based on your own method)


def setup_openai(api_key_path: str = "C:/Users/17284/Desktop/temp/config.yaml"):
    with open(api_key_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


OPENAI_CLIENT = setup_openai()


# Define the response model
class WarrenBuffettSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def create_default_warren_buffett_signal():
    return WarrenBuffettSignal(
        signal="neutral",
        confidence=0.0,
        reasoning="Error in analysis, defaulting to neutral",
    )


def call_llm(
    prompt: list[dict],
    pydantic_model: type[WarrenBuffettSignal] = WarrenBuffettSignal,
    default_factory: Callable[[], WarrenBuffettSignal] = None,
    model: str = "o4-mini-2025-04-16",
    temperature: float = 0.0,
    max_tokens: int = 2000,
) -> WarrenBuffettSignal:
    """
    Call OpenAI LLM and parse the response into the specified Pydantic model.
    Compatible with all GPT series models: prioritizes beta.chat.completions.parse (supports structured output),
    falls back to chat.completions.create with manual JSON parsing.

    Args:
        prompt: List of messages for OpenAI Chat API
        pydantic_model: Pydantic class for parsing the output
        default_factory: Fallback factory function if parsing fails
        model: Name of the model to call
        temperature: Sampling temperature
        max_tokens: Maximum output token count

    Returns:
        Parsed Pydantic instance, or default_factory() (if provided)/None.
    """
    try:
        # Prioritize parse interface that supports structured output
        parse_fn = getattr(OPENAI_CLIENT.beta.chat.completions, "parse", None)
        if parse_fn:
            # For parse, parameter name is max_completion_tokens
            resp = parse_fn(
                model=model,
                messages=prompt,
                # since o4-mini-2025-04-16 does not support max_tokens, we use max_completion_tokens
                # since o4-mini defaulty greedy algorithm, they do not support temperature
                max_completion_tokens=max_tokens,
                response_format=pydantic_model,
            )
            content = resp.choices[0].message.content
        else:
            # Fall back to generic create interface
            resp = OPENAI_CLIENT.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content

    except Exception as e:
        print(f"LLM call failed: {e}")
        raise

    # Finally parse JSON with Pydantic
    try:
        return pydantic_model.model_validate_json(content)
    except Exception as e:
        print(f"Failed to parse response: {e}")
        if default_factory:
            return default_factory()
        return None


# # Test
# if __name__ == "__main__":
#     # Example analysis data
#     analysis_data = """
#     ROE: 15%
#     Debt to Equity: 0.5
#     Current Ratio: 1.5
#     Operating Margin: 20%
#     Revenue Growth: 10%
#     Free Cash Flow: $1B
#     """
#     ticker = "AAPL"
#     # Define the template
#     template = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 "You are a testing assistant. Follow instructions precisely and return valid JSON.",
#             ),
#             (
#                 "human",
#                 """Please analyze the symbol "{ticker}" using the data below:

#     {analysis_data}

#     Return your answer in exactly this JSON format:
#     {{
#     "signal": "buy" | "sell" | "hold",
#     "confidence": float between 0 and 100,
#     "comments": "brief explanation of your decision"
#     }}
#     """,
#             ),
#         ]
#     )

#     prompt_value = template.format_prompt(ticker=ticker, analysis_data=analysis_data)
#     messages = prompt_value.to_messages()
#     prompt = []
#     for m in messages:
#         role = m.type  # this will be "system", "human", or "ai"
#         if role == "human":
#             role = "user"
#         elif role == "ai":
#             role = "assistant"

#         prompt.append({"role": role, "content": m.content})
#     print(prompt)

#     # Analyze AAPL
#     result = call_llm(
#         prompt=prompt,
#         pydantic_model=WarrenBuffettSignal,
#         default_factory=create_default_warren_buffett_signal,
#         model="gpt-4o-2024-08-06",
#         temperature=0.0,
#         max_tokens=2000,
#     )

#     # Print results
#     print("\nInvestment Analysis Results:")
#     print(f"Signal: {result.signal.upper()}")
#     print(f"Confidence: {result.confidence:.1f}%")
#     print(f"Reasoning: {result.reasoning}")
