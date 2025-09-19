

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentType
import os
from dotenv import load_dotenv

load_dotenv()

# 🤖 Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 🔍 SerpAPI search wrapper
search = SerpAPIWrapper()

tools = [
    Tool(
        name="Job Search",
        func=search.run,
        description="Useful for searching job postings online"
    )
]

# 🛠 Initialize agent
job_search_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 🔎 Example: Job postings search
# query = "Generative AI Developer jobs in India September 2025"
query = '''Top 3 links of job postings around Hyderabad for Generative AI Developer with >3 years experience 
        only use linkedin '''
result = job_search_agent.run(query)

print("\n=== Job Search Results ===\n")
print(result)
