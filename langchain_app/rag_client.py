import os
import requests
from typing import List
from pydantic import BaseModel

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv

from langchain_app.faiss_retriever import FaissApiRetriever

# auto-locate and load .env in repo root
load_dotenv(override=True)

AZURE_OPENAI_API_KEY        = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT       = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT     = os.getenv("AZURE_OPENAI_DEPLOYMENT")
OPENAI_API_VERSION          = os.getenv("OPENAI_API_VERSION")
FAISS_SERVICE_URL           = os.getenv("FAISS_SERVICE_URL", "http://my-faiss-service.eastus.azurecontainer.io:8000/search")


PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful AI assistant. Use the following context to answer the question. Cite the sources you used.
If you are unsure of the answer please start the ANSWER with 'I AM NOT COMPLETELY SURE BASED ON THE EVIDENCE /chunks of evidence/, but if I've had to answer'

======== CONTEXT ========
{context}

======== QUESTION ========
{question}

======== ANSWER ========
"""
)
faiss_retriever = FaissApiRetriever(endpoint=FAISS_SERVICE_URL,
                                        k=5,
                                       )
llm = AzureChatOpenAI(
    deployment_name="gpt-4o-mini",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION,
    temperature=0.0
)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=faiss_retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": PROMPT}
)


def answer_question(question: str) -> str:
    return chain.invoke({"query": question})

def return_test():
    return "This was a small test"

if __name__ == "__main__":
    print(answer_question("What is the role of STAG1/STAG2 proteins in differentiation?"))