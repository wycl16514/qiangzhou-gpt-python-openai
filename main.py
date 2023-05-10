import os
import openai
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

os.environ["OPENAI_API_KEY"] = "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"
embeddings = OpenAIEmbeddings()
pinecone.init(
    api_key="d4c93a0a-846b-420f-b72d-6d4d7ed1e584",
    environment="northamerica-northeast1-gcp"
)
index_name = "qiangzhou"
index = Pinecone.from_existing_index(index_name, embeddings)

def get_similiar_docs(query, k=1, score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query, k=k)
  else:
    similar_docs = index.similarity_search(query, k=k)
  return similar_docs

llm = OpenAI()
chain = load_qa_chain(llm, chain_type="stuff")
def get_answer(query):
  similar_docs = get_similiar_docs(query)
  answer = chain.run(input_documents=similar_docs, question=query)
  return answer

from flask import Flask, request, jsonify
from flask import request
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/gpt_query', methods=['POST'])
def easy_ocr():
    query_content = request.json
    query = query_content['query']
    print(f'get query: {query}')
    answer = get_answer(query)
    return jsonify({
        "query": query,
        "answer": answer,
    })

if __name__ == "__main__":
    app.run()