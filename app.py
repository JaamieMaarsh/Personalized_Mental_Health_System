import os
import torch
import faiss
import numpy as np
import gradio as gr
from google.colab import drive
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the Hugging Face token
hf_token = os.getenv('HF_TOKEN')

print(hf_token)  # For debugging, print to check if it's loaded correctly


# Mount Google Drive
drive.mount('/content/drive')

# Define model paths
embedding_model_bge = "BAAI/bge-base-en-v1.5"
save_path_bge = "/content/drive/MyDrive/models/bge-base-en-v1.5"
faiss_index_path = "/content/qa_faiss_embedding.index"
chunked_text_path = "/content/chunked_text_RAG_text.txt"
READER_MODEL_NAME = "google/gemma-2-9b-it"

# Ensure directory exists
os.makedirs(save_path_bge, exist_ok=True)

# Load Sentence Transformer Model
if not os.path.exists(os.path.join(save_path_bge, "config.json")):
    print("Saving model to Google Drive...")
    embedding_model = SentenceTransformer(embedding_model_bge)
    embedding_model.save(save_path_bge)
    print("Model saved successfully!")
else:
    print("Loading model from Google Drive...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = SentenceTransformer(save_path_bge, device=device)

# Load FAISS Index
faiss_index = faiss.read_index(faiss_index_path)
print("FAISS index loaded successfully!")

# Load chunked text
def load_chunked_text():
    with open(chunked_text_path, "r", encoding="utf-8") as f:
        return f.read().split("\n\n---\n\n")

chunked_text = load_chunked_text()
print(f"Loaded {len(chunked_text)} text chunks.")


READER_MODEL_NAME = "google/gemma-2-9b-it"

# Load LLM model and tokenizer
device = 0 if torch.cuda.is_available() else -1
tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME).to(device)
READER_LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=500,
    device=device,
)

def process_query(user_query):
    # Embed the query
    query_embedding = embedding_model.encode(user_query, normalize_embeddings=True)
    query_embedding = np.array([query_embedding], dtype=np.float32)
    
    # Search FAISS index
    k = 5  # Retrieve top 5 relevant docs
    distances, indices = faiss_index.search(query_embedding, k)
    retrieved_docs = [chunked_text[i] for i in indices[0]]
    
    # Construct context
    context = "\nExtracted documents:\n" + "".join([f"Document {i}:::\n{doc}\n" for i, doc in enumerate(retrieved_docs)])
    
    # Define RAG prompt
    prompt_in_chat_format = [
        {"role": "user", "content": f"""
        You are an AI assistant specialized in diagnosing mental disorders in humans.
        Using the information contained in the context, answer the question comprehensively.
        
        The **Diagnosed Mental Disorder** should be only one from the list provided.
        [Normal, Depression, Suicidal, Anxiety, Stress, Bi-Polar, Personality Disorder]
        
        Your response must include:
        1. **Diagnosed Mental Disorder**
        2. **Matching Symptoms**
        3. **Personalized Treatment**
        4. **Helpline Numbers**
        5. **Source Link** (if applicable)
        
        If a disorder cannot be determined, return **Diagnosed Mental Disorder** as "Unknown".
        
        ---
        Context:
        {context}
        
        Question: {user_query}"""},
        {"role": "assistant", "content": ""},
    ]
    
    RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )
    
    # Generate response
    answer = READER_LLM(RAG_PROMPT_TEMPLATE)[0]["generated_text"]
    return answer

# Gradio UI
iface = gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(lines=2, placeholder="Enter your mental health concern here..."),
    outputs=gr.Textbox(label="Personalized Diagnosis"),
    title="Mental Health Diagnosis AI",
    description="Enter your concern and receive AI-powered mental health insights.",
)

# Launch Gradio app
iface.launch(share=True)