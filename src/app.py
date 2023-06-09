import os
import webbrowser
import socket
from flask import Flask, redirect, request, render_template, jsonify, url_for, stream_with_context, Response

from chatbot import generate_answer, chatbot_response, initialize_chatbot

def get_ip_address():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address


# Initialize the chatbot
chain, retriever, memory, provide_sources, chain_type = None, None, None, None, None

app = Flask(__name__)

@app.route("/start", methods=["GET", "POST"])
def start():
    if request.method == "POST":
        global chain, retriever, memory, chain_type, provide_sources
        # Get the user's input from the form
        repo_url_without_subdir = request.form["repo_url"]
        print(f"repo url in request without subdirectory: {repo_url_without_subdir}")
        subdirectory = request.form["subdirectory"]
        if subdirectory != "None" and subdirectory != "":
            print(f"subdirectory: {subdirectory}")
            repo_url = f"{repo_url_without_subdir}/{subdirectory}"
            print(f"repo url in request with subdirectory: {repo_url}")
        else:
            print("No subdirectory")
            repo_url = repo_url_without_subdir
        mem_window_k = request.form["mem_window_k"]
        temperature = request.form["temperature"]
        model_name = request.form["model_name"]
        top_k = request.form["top_k"]
        update = request.form["update"]
        alpha = request.form["alpha"]
        vector_store = request.form["vector_store"]
        compress = request.form["compress"]
        model_name_compressor = request.form["model_name_compressor"]
        provide_sources = request.form["provide_sources"]
        chain_type = request.form["chain_type"]

        # Override sources setting because it is not yet implemented ! REMOVE THIS LINE WHEN IMPLEMENTED !!!
        provide_sources = "false"

        print(f"repo url in request (complete): {repo_url}")

        # initiazlize the chatbot and store the variables in the global scope
        
        chain, retriever, memory = initialize_chatbot(model_name, top_k, temperature, mem_window_k, alpha, repo_url, subdirectory, update, chain_type, vector_store, compress, model_name_compressor, provide_sources)
        # if chain_type == "conv_retr_chain":
        #     memory = []

        print(f"memory: \n\t{memory}")

        return redirect(url_for("index"))
    return render_template("start_page.html")


# Route for handling the chatbot
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text_input = request.form["text_input"]

        def generate(text_input):
            for token in chatbot_response(text_input, generate_answer, memory, chain, retriever, provide_sources, chain_type):
                print(token)
                yield token

        return Response(stream_with_context(generate(text_input)), content_type='text/plain')

    return render_template("index.html")


@app.route("/clear_memory", methods=["POST"])
def clear_memory():
    memory.clear()
    return jsonify(status="success")


if __name__ == "__main__":
    ip_address = get_ip_address()
    url = f"http://{ip_address}:5000/start"
    webbrowser.open_new(url)
    app.run(
        # debug=True,
        # host=ip_address
        host="0.0.0.0"  # TODO: security issue
        )
