# ✅ rag_chain.py - Updated version with speech + translation support

import os
import pyttsx3
from dotenv import load_dotenv
from googletrans import Translator

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA

# ✅ Load environment variables from .env
if not load_dotenv(dotenv_path=".env"):
    print("⚠️ .env file not found. Ensure it exists.")
    exit(1)

# ✅ Validate OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key or not (openai_api_key.startswith("sk-") or openai_api_key.startswith("sk-proj-")):
    print("❌ ERROR: OPENAI_API_KEY is missing or invalid in your .env file.")
    exit(1)
print("🔐 OpenAI API Key loaded successfully.")

# ✅ Set up translator and speech engine
translator = Translator()
speech_engine = pyttsx3.init()
speech_engine.setProperty('rate', 160)  # speaking speed

# ✅ Load FAISS DB and setup RAG pipeline
def load_rag_chain():
    try:
        print("📦 Loading embedding model...")
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        print("🗃️ Loading FAISS vector DB...")
        db = FAISS.load_local(
            folder_path="form_vector_db",
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        print("✅ FAISS vector DB loaded.")

        retriever = db.as_retriever(search_kwargs={"k": 3})

        print("🧠 Initializing LLM...")
        llm = OpenAI(openai_api_key=openai_api_key, temperature=0.3)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        print("✅ RAG chain is ready.")
        return qa

    except Exception as e:
        print(f"❌ Failed to load RAG components: {e}")
        exit(1)

# ✅ Function to speak response

def speak(text, lang='en'):
    try:
        if lang != 'en':
            translated = translator.translate(text, dest=lang)
            text = translated.text
        print("🗣️ Speaking Response...")
        speech_engine.say(text)
        speech_engine.runAndWait()
    except Exception as e:
        print(f"⚠️ Speech error: {e}")


# ✅ CLI Loop
if __name__ == "__main__":
    qa = load_rag_chain()

    print("\n📋 RAG System Ready — Ask Questions About Your Forms!")
    print("💡 Example: What is needed for PPF withdrawal?")
    print("💡 Example: How to update KYC?")
    print("(Type 'exit' to quit)")

    while True:
        query = input("\n🧠 Ask: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting. Have a nice day!")
            break

        try:
            result = qa(query)
            answer = result['result']

            print("\n💬 Answer:\n", answer)
            print("\n📄 Source(s):")
            for doc in result['source_documents']:
                print(f"→ {doc.metadata.get('form_name', 'Unknown')}")
            print("\n" + "-" * 60)

            # Speak answer aloud in English (or change to 'hi' for Hindi, etc.)
            speak(answer, lang='en')

        except Exception as e:
            print(f"⚠️ Error while processing query: {e}")
