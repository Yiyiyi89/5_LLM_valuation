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
    prompt: Any,
    pydantic_model=WarrenBuffettSignal,
    default_factory=None,
    model: str = "gpt-4o-2024-08-06",
    temperature: float = 0.0,
    max_tokens: int = 2000,
) -> WarrenBuffettSignal:
    """
    Analyze a stock using LLM based on provided financial metrics.

    Args:
        ticker: Stock symbol to analyze
        analysis_data: String containing financial metrics and analysis data
        model: OpenAI model to use
        temperature: Temperature for response generation
        max_tokens: Maximum tokens in the response

    Returns:
        InvestmentAnalysis: Analysis result containing signal, confidence, and comments
    """

    # Call ChatGPT
    try:
        # 1) invoke the API via the chat.completions.create method
        resp = OPENAI_CLIENT.beta.chat.completions.parse(
            model=model,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=pydantic_model,
        )

    except Exception as e:
        print(f"LLM call failed: {e}")
        raise

    # Parse JSON response
    content = resp.choices[0].message.content
    try:
        return pydantic_model.model_validate_json(content)
    except Exception as e:
        print(f"Failed to parse response: {e}")
        return default_factory


# Test
if __name__ == "__main__":
    # Example analysis data
    analysis_data = """
    ROE: 15%
    Debt to Equity: 0.5
    Current Ratio: 1.5
    Operating Margin: 20%
    Revenue Growth: 10%
    Free Cash Flow: $1B
    """
    ticker = "AAPL"
    # Define the template
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a testing assistant. Follow instructions precisely and return valid JSON.",
            ),
            (
                "human",
                """Please analyze the symbol "{ticker}" using the data below:

    {analysis_data}

    Return your answer in exactly this JSON format:
    {{
    "signal": "buy" | "sell" | "hold",
    "confidence": float between 0 and 100,
    "comments": "brief explanation of your decision"
    }}
    """,
            ),
        ]
    )

    prompt_value = template.format_prompt(ticker=ticker, analysis_data=analysis_data)
    messages = prompt_value.to_messages()
    prompt = []
    for m in messages:
        role = m.type  # this will be "system", "human", or "ai"
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"

        prompt.append({"role": role, "content": m.content})
    print(prompt)

    # Analyze AAPL
    result = call_llm(
        prompt=prompt,
        pydantic_model=WarrenBuffettSignal,
        default_factory=create_default_warren_buffett_signal,
        model="gpt-4o-2024-08-06",
        temperature=0.0,
        max_tokens=2000,
    )

    # Print results
    print("\nInvestment Analysis Results:")
    print(f"Signal: {result.signal.upper()}")
    print(f"Confidence: {result.confidence:.1f}%")
    print(f"Reasoning: {result.reasoning}")
