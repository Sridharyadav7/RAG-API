from fastapi import FastAPI, Query, HTTPException
from langchain.chains import RetrievalQA
import google.generativeai as genai
from langchain.llms.base import LLM
from typing import Optional, List
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient
from langchain.prompts import PromptTemplate 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import asyncio


load_dotenv()
app = FastAPI(title="RAG Medical API")

# Global variables to store components
rag_chain = None

@app.on_event("startup")
async def startup_event():
    global rag_chain

    try:
        print("Initializing RAG components...")
        
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
        
        # Initialize Ollama with better parameters
        llm = ChatOpenAI(
            model_name="gpt-4",  # or "gpt-4"
            temperature=0.3,
            max_tokens=1024,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize Pinecone
        pinecone_client = PineconeClient(api_key=PINECONE_API_KEY)
        
        # Create vector store
        vectorstore = PineconeVectorStore(
            embedding=embedding_model, 
            index_name=PINECONE_INDEX_NAME
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(search_type="similarity", k=1)
        
        # Create a proper prompt template
        prompt_template = """
        You are a helpful medical assistant providing accurate and complete information.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use proper formatting with complete sentences and paragraphs for readability.

        Context: {context}

        Question: {question}

        Helpful Answer:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create RAG chain with proper prompt
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff",
            retriever=retriever, 
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("RAG components initialized successfully")
        
    except Exception as e:
        print(f"Error initializing RAG components: {str(e)}")
        # Print stack trace for more debugging info
        import traceback
        traceback.print_exc()


@app.get("/ask")
async def ask_question(query: str = Query(..., description="Your medical question")):
    global rag_chain
    
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG system not initialized. Please check server logs.")
    
    try:
        # Run the LLM query in a thread pool to not block the event loop
        result = await asyncio.to_thread(rag_chain.invoke, {"query": query})
        
        # Clean and format the answer
        raw_answer = result.get("result", "No answer generated")
        cleaned_answer = raw_answer.replace("\n", " ").replace("*", "").replace("•", "").strip()

        
        return cleaned_answer
        
    except Exception as e:
        import traceback
        err_details = traceback.format_exc()
        print(f"Error in /ask endpoint: {str(e)}\n{err_details}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "ok", "rag_initialized": rag_chain is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
