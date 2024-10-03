import pymupdf as pdf
import pymupdf4llm as pdfllm
from llama_index.core import VectorStoreIndex, Document, SimpleDirectoryReader
import chromadb
from sentence_transformers import SentenceTransformer

documents = SimpleDirectoryReader("ChatWithPdf/data").load_data()

#Initialize SentenceTransformer model for creating embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

#Index using llamaindex
index = VectorStoreIndex.from_documents(documents)

#Extract text and store it in chunks
chunks = [doc.get_text() for doc in documents]

#Generate embeddings for each document chunk using the embedding model
embeddings = [model.encode(chunk) for chunk in chunks]

#3. Store the embeddings in ChromaDB
#Initialize the client
chroma_client = chromadb.Client()

#Create a collection for storage
collection = chroma_client.create_collection("document_chunks")

#Loop to add all document chunks to the collection
for i, chunk in enumerate(chunks):
    collection.add(
        document = [chunk],
        embeddings = [embeddings[i]],
        metadatas = [{"doc_id": i}] 
    )

#4. Query and retrieve results using semantic search
#Example query
user_query = "How did it all start?"

#4.1 Convert the query to an embedding
query_embedding = model.encode(user_query)

#4.2 Parse through chromadb for the most relevant chunks based on the user query
results = collection.query(query_embeddings=[query_embedding], n_results=2)

#4.3 Get the most relevant document chunks based on similarity
retrieved_chunks = [result['document'][0] for result in results['document']]

#4.4 Use Llamaindex to further refine and query the chunks
final_response = index.query(user_query, document=[Document(chunk) for chunk in retrieved_chunks])
print(final_response)
