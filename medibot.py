from flask import Flask, request, jsonify, render_template
import os
import csv
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline

app = Flask(__name__)

# === Configuration ===
PDF_VECTORSTORE_PATH = "vectorstore/db_faiss"
DOCTOR_CSV_PATH = "doctors.csv"

CUSTOM_PROMPT_TEMPLATE = """


Context: {context}
Question: {question}

Thanks for your queries. Please wait
"""


# === Load Vectorstore from FAISS ===
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.load_local(PDF_VECTORSTORE_PATH, embedding_model, allow_dangerous_deserialization=True)

# === Load LLM (Falcon or similar) ===
def load_llm():
    model_name = "tiiuae/falcon-rw-1b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pipe)

# === Load doctor list with condition mappings ===
def load_doctors():
    doctors = []
    with open("doctors.csv", newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            doctors.append(row)
    return doctors

DOCTOR_LIST = load_doctors()


def match_doctor_by_topic(user_input):
    input_lower = user_input.lower()
    for doc in DOCTOR_LIST:
        if any(keyword.lower() in input_lower for keyword in [doc['Specialty'], doc.get('Condition', '')]):
            return f"You can consult {doc['Name']}, a specialist in {doc['Specialty']}."
    return None



def log_chat(sender, message):
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open("chat_history.txt", "a", encoding="utf-8") as f:
        f.write(f"[{time_str}] {sender}: {message}\n")


# === LangChain setup ===
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = load_llm()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": PromptTemplate(
            template=CUSTOM_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
    }
)


# === Load doctors ===
DOCTOR_LIST = load_doctors()

# === Flask Routes ===
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")

    # 1. Get the PDF-based answer using LangChain
    response = qa_chain.invoke({'query': user_message})
    bot_reply = response["result"]

    # 2. Add doctor recommendation, if any
    doctor_ref = match_doctor_by_topic(user_message)
    if doctor_ref:
        bot_reply += f"\n\n👨‍⚕️ {doctor_ref}"  # ✅ APPEND instead of REPLACE

    # 3. Save to chat log
    log_chat("User", user_message)
    log_chat("Bot", bot_reply)

    return jsonify({"response": bot_reply})


if __name__ == "__main__":
    import os
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or os.environ.get("FLASK_ENV") != "development":
        app.run(debug=True)

