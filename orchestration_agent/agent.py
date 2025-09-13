from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from .prompt import instruction
from .tools.get_plumber_electrician_details import get_nearest_trades
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

root_agent = LlmAgent(
   # A unique name for the agent.
   name="orchestration_agent",
   # The Large Language Model (LLM) that agent will use.
   # Please fill in the latest model id that supports live from
   # https://google.github.io/adk-docs/get-started/streaming/quickstart-streaming/#supported-models
   model=LiteLlm(model="command-a-03-2025"),  # for example: model="gemini-2.0-flash-live-001" or model="gemini-2.0-flash-live-preview-04-09"
   # A short description of the agent's purpose.
   description="AI orchestration agent that can find nearest plumbers and electricians using Google Places API",
   # Instructions to set the agent's behavior.
   instruction=instruction,
   tools=[get_nearest_trades]
)