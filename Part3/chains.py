import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


# summarization chain
def get_summary_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    template = """Retrieve relevant information and summarize it concisely for the user.
    Context: {context}
    User Query: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = {"context": retriever, "question": lambda x: x} | prompt | llm | StrOutputParser()

    return chain


# code generation
def get_code_chain():
    template = """You are an expert Python programmer. Write clean, commented code for the following request.
    Request: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    return chain


# web search chain
def get_web_search_chain():
    search_tool = TavilySearch(max_results=3)

    def run_search(query):
        """Helper to run search and format results safely."""
        try:

            raw_response = search_tool.invoke(query)
            
            if isinstance(raw_response, dict) and "results" in raw_response:
                results = raw_response["results"]
            elif isinstance(raw_response, list):
                results = raw_response
            else:

                return f"Unexpected search format: {type(raw_response)}"


            context_str = ""
            for res in results:

                url = res.get('url', 'No URL')
                content = res.get('content', 'No Content')
                context_str += f"Source: {url}\nContent: {content}\n\n"
            
            return context_str if context_str else "No results found."
        
        except Exception as e:
            print(f"Error details: {e}")
            return f"Error searching the web: {e}"

    template = """You are a helpful assistant. Answer the user's question based strictly on the search results provided below.
    
    Search Results:
    {context}
    
    User Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": run_search, "question": lambda x: x}
        | prompt 
        | llm 
        | StrOutputParser()
    )
    return chain