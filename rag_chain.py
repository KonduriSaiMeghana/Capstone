# rag_chain.py

import os
from dotenv import load_dotenv

# ✅ Step 0: Load .env and check presence
if not load_dotenv(dotenv_path=".env"):
    print("⚠️ .env file not found in the current directory. Make sure it exists.")
    exit(1)

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA

# ✅ Step 1: Validate OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key or not (openai_api_key.startswith("sk-") or openai_api_key.startswith("sk-proj-")):
    print("❌ ERROR: OPENAI_API_KEY is missing or invalid in your .env file.")
    exit(1)

print("🔐 OpenAI API Key loaded successfully.")

# ✅ Step 2: Load FAISS DB and setup chain
def load_rag_chain():
    try:
        # Load embeddings
        print("📦 Loading embedding model...")
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Load vector DB
        print("🗃️ Loading FAISS vector DB...")
        db = FAISS.load_local(
            folder_path="form_vector_db",
            embeddings=embedding_model,
            allow_dangerous_deserialization=True  # Only use if trusted
        )
        print("✅ FAISS vector DB loaded.")

        # Retriever with top 3 chunks
        retriever = db.as_retriever(search_kwargs={"k": 3})

        # LLM setup
        print("🧠 Initializing LLM...")
        llm = OpenAI(openai_api_key=openai_api_key, temperature=0.3)

        # Chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        print("✅ RAG chain is ready.\n")
        return qa

    except Exception as e:
        print(f"❌ Failed to load RAG components: {e}")
        exit(1)

# ✅ Step 3: Ask questions
if __name__ == "__main__":
    qa = load_rag_chain()

    print("📋 RAG System Ready — Ask Questions About Your Forms!")
    print("💡 Example: What is needed for PPF withdrawal?")
    print("💡 Example: How to update KYC?\n(Type 'exit' to quit)\n")

    while True:
        query = input("🧠 Ask: ").strip()
        if query.lower() in ['exit', 'quit']:
            print("👋 Exiting. Have a nice day!")
            break

        try:
            result = qa(query)
            print("\n💬 Answer:\n", result['result'])

            print("\n📄 Source(s):")
            for doc in result['source_documents']:
                print(f"→ {doc.metadata.get('form_name', 'Unknown')}")
            print("\n" + "-" * 60)

        except Exception as e:
            print(f"⚠️ Error while processing query: {e}")
