from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.load import dumps, loads


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


# simple RAG chain
def get_simple_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = {"context": retriever, "question": lambda x: x} | prompt | llm | StrOutputParser()
    return chain



# RAG Fusion funcs
def reciprocal_rank_fusion(results: list[list], k=60):
    """Reciprocal Rank Fusion algorithm to re-rank documents."""
    
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in fused_scores.items()
    ]
    reranked_results.sort(key=lambda x: x[1], reverse=True)
    
    return [x[0] for x in reranked_results]


# RAG fusion chain
def get_fusion_chain(vectorstore):
    
    query_gen_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant that generates multiple search queries based on a single input query. \n"
        "Generate 4 search queries related to: {question}. These queries should explore different aspects of the original asked.\n"
        "Output (4 queries):"
    )
    
    # create a generator chain
    generate_queries = query_gen_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    
    retriever = vectorstore.as_retriever()
    
    def retrieval_node(question):

        # generate the fusion queries for the incoming question
        queries = generate_queries.invoke({"question": question})
        
        # print(f"Generated Queries: {queries}")
        
        # retrieve files based on those queries
        
        results = [retriever.invoke(q) for q in queries]
        return reciprocal_rank_fusion(results)

    
    # make the final chain
    final_template = """Answer the question based on the multiple retrieved contexts:
    {context}
    
    Question: {question}
    """
    
    final_prompt = ChatPromptTemplate.from_template(final_template)
    
    chain = {"context": retrieval_node, "question": lambda x: x} | final_prompt | llm | StrOutputParser()
    
    return chain