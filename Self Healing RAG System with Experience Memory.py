# Self Healing RAG Agent with Experience Memory

import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.documents import Document
import json

llm = ChatOpenAI(
    model="gpt-4o-mini", temperature = 0) # I want to minimize hallucination - temperature = 0 makes the model output more deterministic 

evaluation_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# Our Embedding Model - has to also be compatible with the LLM
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)


pdf_path = "Stock_Market_Performance_2024.pdf"


# Safety measure I have put for debugging purposes :)
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path) # This loads the PDF

# Checks if the PDF is there
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

# Chunking Process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


pages_split = text_splitter.split_documents(pages) # We now apply this to our pages

persist_directory = r"C:\Users\Rahul N\Downloads\selfhealing rag"
collection_name = "stock_market"

# If our collection does not exist in the directory, we create using the os command
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


try:
    # Here, we actually create the chroma database using our embeddigns model
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store!")
    
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise

memory_collection = Chroma(
    collection_name="experience_memory",
    embedding_function=embeddings,
    persist_directory=persist_directory
)


# Now we create our retriever 
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # K is the amount of chunks to return
)

@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Stock Market Performance 2024 document.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the Stock Market Performance 2024 document."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)

@tool
def memory_search(query: str) -> str:
    """
    Search previous successful retrieval experiences.
    """

    similar_questions = memory_collection.similarity_search(
        query,
        k=3
    )

    if not similar_questions:
        return "No relevant past experiences."

    results = []

    for doc in similar_questions:

        final_answer = doc.metadata.get("final_answer", "")
        best_query = doc.metadata.get("best_query", "")
        feedback = doc.metadata.get("feedback", "")
        score = doc.metadata.get("score", "")

        results.append(
            f"""

        Retrieved Similar Question:
        {doc.page_content}

        Successful Retrieval Query:
        {best_query}

        Final Answer:
        {final_answer}

        Evaluator Feedback:
        {feedback}

        Final Score:
        {score}

    """
        )

    return "\n\n".join(results)

tools = [retriever_tool, memory_search]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    
    messages: Annotated[Sequence[BaseMessage], add_messages]
    score: float
    feedback: str
    iteration: int


def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
You are a self-improving retrieval agent.

Available tools:

1. memory_search
   Search previous successful experiences.

2. retriever_tool
   Search the PDF.

Always search memory first.

Use past successful retrieval strategies whenever relevant.

Use evaluator feedback to improve retrieval quality.
"""


# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    feedback = state.get('feedback', '')
    feedback_prompt = f""" Use the following evaluator feedback to improve retrieval and answering: {feedback} """
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + [SystemMessage(content=feedback_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}

tool_node = ToolNode(tools)

def evaluator(state: AgentState):

    question = state["messages"][0].content

    final_answer = ""

    for msg in reversed(state["messages"]):

        if msg.type == "ai":
            final_answer = msg.content
            break

    prompt = f"""
    You are a retrieval evaluator.

    Question:
    {question}

    Final Answer:
    {final_answer}

    Evaluate:

    1. Relevance
    2. Completeness
    3. Ability to answer the question

    Return JSON:

    {{
      "score": 0-10,
      "feedback": "specific feedback"
    }}
    """

    result = evaluation_llm.invoke(prompt)

    try:
        evaluation = json.loads(result.content)

        score = float(evaluation["score"])

        feedback = evaluation["feedback"]

    except:
        score = 0
        feedback = "Evaluation failed."

    return {
        "score": score,
        "feedback": feedback,
        "iteration": state["iteration"] + 1
    }

THRESHOLD = 8
MAX_ITERATIONS = 5

def evaluator_router(state: AgentState):

    score = state["score"]

    iteration = state["iteration"]

    print(
        f"Iteration={iteration} "
        f"Score={score}"
    )

    if score >= THRESHOLD:
        return "finish"

    if iteration >= MAX_ITERATIONS:
        return "finish"

    return "retry"

def get_last_retrieval_query(state: AgentState):

    for msg in reversed(state["messages"]):

        if hasattr(msg, "tool_calls") and msg.tool_calls:

            for tool_call in msg.tool_calls:

                if tool_call["name"] == "retriever_tool":

                    return tool_call["args"].get("query", "")

    return ""

def store_experience(state: AgentState):

    if state["score"] < THRESHOLD:
        return {}

    question = state["messages"][0].content

    feedback = state["feedback"]

    final_answer = ""

    best_query = ""

    for msg in reversed(state["messages"]):

        if msg.type == "ai":
            final_answer = msg.content
            break
        
    best_query = get_last_retrieval_query(state)

    memory_collection.add_documents(
        [
            Document(
                page_content = question,
                metadata = {
                    "final_answer": final_answer,
                    "best_query": best_query,
                    "feedback": feedback,
                    "score": state['score']
                }
            )
        ]
    )

    print("Stored successful experience.")

    return {}


graph = StateGraph(AgentState)
graph.add_node("retriever_agent", call_llm)
graph.add_node("tools", tool_node)
graph.add_node("evaluator", evaluator)
graph.add_node("store_experience", store_experience)

graph.add_conditional_edges(
    "retriever_agent",
    should_continue,
    {True: "tools", False: "evaluator"}
)

graph.add_conditional_edges(
    "evaluator",
    evaluator_router,
    {"finish": "store_experience", "retry": "retriever_agent"}
)

graph.add_edge("store_experience", END)
graph.add_edge("tools", "retriever_agent")
graph.set_entry_point("retriever_agent")

rag_agent = graph.compile()


def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        result = rag_agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content = user_input
                )
            ],
            "score": 0,
            "feedback": "",
            "iteration": 0
        }
        )
        
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


running_agent()