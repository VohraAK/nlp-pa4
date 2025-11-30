"""Actual routing mechanism for assigning chains, and dynamic chain creation as necessary..."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from chains import get_summary_chain, get_code_chain, get_web_search_chain
from rag import get_fusion_chain

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def generate_and_validate_tool(task_description):
    # dynamic chain creation logic

    # tool function creator
    gen_prompt = ChatPromptTemplate.from_template(
        "Write a Python function using @tool decorator for: {task}. Return ONLY code."
    )
    
    code_chain = gen_prompt | llm | StrOutputParser()
    generated_code = code_chain.invoke({"task": task_description})
    
    # validation chain for tool val
    val_prompt = ChatPromptTemplate.from_template(
        "Is this Python code safe and syntactically correct? Return 'VALID' or 'INVALID'.\nCode: {code}"
    )
    validator_chain = val_prompt | llm | StrOutputParser()
    validation = validator_chain.invoke({"code": generated_code})
    
    if "VALID" in validation:
        # did not do exec here, just ran the output...
        return f"SUCCESS: Created and validated new tool for '{task_description}'."
    return "ERROR: Generated code failed validation."

def route_and_execute(query, vectorstore):

    # chain routing logic
    
    # routing chain -> get correct chain for user query / task
    router_template = """Classify the user query into one category:
    - SUMMARY: Requests to summarize specific documents/topics.
    - CODE: Requests to generate source code.
    - WEB: Requests requiring internet search.
    - CREATE: Requests for a specific new tool/utility not covered above.
    - GENERAL: Ambiguous or general questions.
    
    Return ONLY the category name.
    Query: {question}
    """
    
    category = (ChatPromptTemplate.from_template(router_template) | llm | StrOutputParser()).invoke({"question": query}).strip().upper()
    
    # choose a chain based on the llm output
    response = ""
    if "SUMMARY" in category:
        chain = get_summary_chain(vectorstore)
        response = chain.invoke(query)
        
    elif "CODE" in category:
        chain = get_code_chain()
        response = chain.invoke(query)
        
    elif "WEB" in category:
        chain = get_web_search_chain()
        response = chain.invoke(query)
        
    elif "CREATE" in category:
        # Trigger Section 6.2 Logic
        response = generate_and_validate_tool(query)
        
    else:
        # Fallback to RAG Fusion
        chain = get_fusion_chain(vectorstore)
        response = chain.invoke(query)
        
    return response, category

