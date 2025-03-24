from flask import Flask, request, jsonify, render_template, send_from_directory
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_community.llms import Ollama  # Correct import for Ollama LLM
import os

app = Flask(__name__)

os.makedirs("uploads", exist_ok=True)
os.makedirs("indexes", exist_ok=True)
os.makedirs("static", exist_ok=True)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

try:
    # ✅ Fix: Correct Ollama model name
    llm = Ollama(model="llama3")  # Change model if needed (run `ollama list`)

    # ✅ Fix: Ensure embedding model works correctly
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # ✅ Fix: Properly register LLM in LlamaIndex settings
    Settings.llm = llm
    Settings.embed_model = embed_model

except Exception as e:
    print(f"❌ Error initializing LLM or embedding model: {e}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        print(f"📂 File saved: {file_path}")

        # ✅ Debug: Load documents safely
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        print(f"📄 Loaded Documents: {documents}")
        if not documents:
            return jsonify({"error": "Failed to extract text from document"}), 400

        # ✅ Debug: Ensure index is created properly
        index = VectorStoreIndex.from_documents(documents)
        index_dir = os.path.join("indexes", file.filename)
        index.storage_context.persist(persist_dir=index_dir)

        print(f"✅ Index created and stored at: {index_dir}")

        return jsonify({"message": "File uploaded and processed successfully", "index_dir": file.filename}), 200
    except Exception as e:
        print(f"❌ Error in /upload: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        query_text = data.get('query')
        index_dir = data.get('index_dir')

        if not query_text or not index_dir:
            return jsonify({"error": "Missing query or index_dir"}), 400

        index_path = os.path.join("indexes", index_dir)
        if not os.path.exists(index_path):
            print(f"❌ Error: Index directory not found: {index_path}")
            return jsonify({"error": "Index directory not found"}), 400

        print(f"🔍 Loading index from: {index_path}")
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)

        query_engine = index.as_query_engine()
        response = query_engine.query(query_text)

        print(f"💬 Query: {query_text}")
        print(f"📝 Response: {response}")

        return jsonify({"response": str(response)}), 200
    except Exception as e:
        print(f"❌ Error in /query: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
