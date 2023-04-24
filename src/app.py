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
chain, retriever, memory, provide_sources = None, None, None, None

app = Flask(__name__)

@app.route("/start", methods=["GET", "POST"])
def start():
    if request.method == "POST":
        # Get the user's input from the form
        repo_url_without_subdir = request.form["repo_url"]
        print(f"repo url in request without subdirectory: {repo_url_without_subdir}")
        subdirectory = request.form["subdirectory"]
        if subdirectory != "None":
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
        global provide_sources
        provide_sources = request.form["provide_sources"]

        print(f"repo url in request complete: {repo_url}")

        # chain, retriever, memory = initialize_chatbot(model_name, top_k, temperature, mem_window_k, alpha, repo_url, subdirectory, update)

        # initiazlize the chatbot and store the variables in the global scope
        global chain, retriever, memory
        chain, retriever, memory = initialize_chatbot(model_name, top_k, temperature, mem_window_k, alpha, repo_url, subdirectory, update, vector_store, compress, model_name_compressor, provide_sources)

        return redirect(url_for("index"))
    return render_template("start_page.html")


# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         text_input = request.form["text_input"]
#         response = chatbot_response(
#             text_input, 
#             generate_answer,
#             memory,
#             chain,
#             retriever,
#             provide_sources
#             )
#         return jsonify(response=response)

#     return render_template("index.html")


# first attempt at streaming
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text_input = request.form["text_input"]

        def generate(text_input):
            for token in chatbot_response(text_input, generate_answer, memory, chain, retriever, provide_sources):
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
    app.run(debug=True, host=ip_address)
