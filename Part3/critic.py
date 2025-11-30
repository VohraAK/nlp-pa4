"""Critic chain logic..."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from rag import get_simple_rag_chain, get_fusion_chain
from router import route_and_execute

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

critic_template = """
You are an AI Critic. Evaluate these three responses to the query: "{query}"

[System 1: Standard RAG]
{rag_ans}

[System 2: RAG-Fusion]
{fusion_ans}

[System 3: RAG-Fusion with Routing (Category: {category})]
{routing_ans}

Evaluation Task:
1. Assign a generic score (1-10) to each system based on accuracy and relevance.
2. Which system provided the highest reward/value? Why?
3. Did Reciprocal Rank Fusion (used in Systems 2 & 3) improve the quality compared to System 1?

Format the output as a concise report.
"""

prompt = ChatPromptTemplate.from_template(critic_template)
critic_chain = prompt | llm | StrOutputParser()


def run_critic(query, vectorstore):
    """
    Executes the LLM Critic Evaluation (Section 6.3).
    Compares RAG, RAG-Fusion, and Routing.
    Prints results to TERMINAL.
    """
    print(f"\n\n{'='*50}\nRUNNING CRITIC EVALUATION FOR: '{query}'\n{'='*50}")

    # run all 3 systems 
    print("1. Generating Standard RAG response...")
    rag_ans = get_simple_rag_chain(vectorstore).invoke(query)
    
    print("2. Generating RAG-Fusion response...")
    fusion_ans = get_fusion_chain(vectorstore).invoke(query)
    
    print("3. Generating Routed response...")
    routing_ans, category = route_and_execute(query, vectorstore)

    # get response
    report = critic_chain.invoke({
        "query": query,
        "rag_ans": rag_ans, 
        "fusion_ans": fusion_ans,
        "routing_ans": routing_ans,
        "category": category
    })
    
    # print to terminal
    print("\n" + "="*50)
    print("FINAL CRITIC REPORT")
    print("="*50)
    print(report)
    print("="*50 + "\n")