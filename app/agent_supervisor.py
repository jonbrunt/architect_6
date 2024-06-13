# Importing necessary modules and classes
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

import functools
import operator

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from langgraph.graph import StateGraph, END

# Defining tools
tavily_tool = TavilySearchResults(max_results=5)

# This executes code locally, which can be unsafe
python_repl_tool = PythonREPLTool()

# System prompt for the supervisor agent
# ===================FILL IN: Members========================
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

# Options for the supervisor to choose from
# ===================FILL IN: Options========================

# Function definition for OpenAI function calling
# ===================FILL IN: Supervisor Function Def========================

# Prompt for the supervisor agent
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

# Initializing the language model
llm = ChatOpenAI(model="gpt-4o")

# Creating the supervisor chain
# ===================FILL IN: Supervisor Chain========================

# Defining a typed dictionary for agent state
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str

# Function to create an agent
def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

# Function to create an agent node
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

# Creating agents and their corresponding nodes

# ===================FILL IN: Reserach Agent Creation========================

# ===================FILL IN: Review Agent Creation==========================

# ===================FILL IN: Coder Agent Creation===========================

# ===================FILL IN: QA Tester Agent Creation========================


# Defining the workflow using StateGraph
# ===================FILL IN: Add noes to workflow========================

# Adding edges to the workflow
for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")

# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
# ===================FILL IN: Define Graph========================

# Finally, add entry point
# ===================FILL IN: Entry Point=======================

# Compile the workflow into a graph
graph = workflow.compile()