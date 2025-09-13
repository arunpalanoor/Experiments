# Import necessary library

import os
import gradio as gr

from dotenv import load_dotenv
import openai
from agents import Agent, Runner, WebSearchTool, trace, OpenAIChatCompletionsModel, handoff, function_tool
from openai import OpenAI

import asyncio

# Get API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define LLM client
llm_model = "gpt-4o-mini"

# Agent definition
news_agent_instructions = '''
You are a financial market research assistant focused on Indian public companies like Infosys, TCS, etc.

When asked about a company’s earnings or financial outlook:
1. Use the web search tool to find the latest news from reliable sources like Economic Times, Moneycontrol, CNBC-TV18, etc.
2. Summarize the upcoming earnings release date, time, and any press or analyst calls.
3. Provide market expectations:
   - Profit (PAT) YoY change
   - Revenue estimates
   - Operating margins
   - Notable trends (e.g., wage hikes, GenAI initiatives)
4. Highlight key focus areas (guidance, TCV, BFSI outlook).
5. Wrap up with a 2–3 sentence summary and add source links in markdown.

Format your response like this:

---

**🗓️ Earnings Date & Time**
<...>

---

**📊 General Market Expectations**
- Profit: ...
- Revenue: ...
- Margins: ...
- Key mentions: ...

---

**🔍 Key Monitorables**
- Guidance
- Deal pipeline
- Segment performance
- Tech initiatives

---

**🧭 Summary**
<Market tone summary>

---

**📚 Sources**
- [Title](https://example.com)
'''

news_agent = Agent(
    name="News Web searcher",
    handoff_description="Specialist agent for web or internet search questions",
    instructions=news_agent_instructions,
    model=llm_model,
    handoffs=[],
    tools=[WebSearchTool(search_context_size="low")],
)

# Run Agent
async def main():
    result = await Runner.run(
                news_agent,
                "What is the latest news about Infosys Share buyback?",
            )

    print(result.final_output)

asyncio.run(main())
