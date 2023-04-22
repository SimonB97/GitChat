# GitChat

GitChat is a chatbot that helps users to get insights into GitHub repositories. It is designed to provide interactive assistance and support for users working with files in a GitHub repository. By integrating chat capabilities, the chatbot can help users  understand code and answer questions related to the project. The chatbot utilizes OpenAI LLMs, Langchain and Pinecone

## Installation

To install GitChat, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies using: \
    `pip install -r requirements.txt`
3. Run the chatbot using `python flask_app.py`.

## Prerequisites

- copy `.env.example` to `.env` and fill in the required fields.

## Usage

To use GitChat, follow these steps:

1. Start the application by running `python flask_app.py`.
2. This will open a web browser and navigate to the start page (http://{ip_address}:5000/start - ip_address is the ip address of the machine running the application, will be automatically detected)
3. Fill in the required fields on the start page and click "Submit" to initialize the chatbot.
4. Once the chatbot is initialized, you will be redirected to the main chat interface.

## Features

- Question Answering powered by OpenAI LLMs (and others to come)
- Chat History (AI has short term memory, like ChatGPT)
- Text sections of files relevant to the question are passed to the AI as context
    - Questions are answered based on the relevant context
- Easily Customizable Options for LLM, Vectorstore and retrieval method

#### Configuration options

- *LLMs*
    - Models:
        - OpenAI-models: GPT-3.5-Turbo, GPT-4
        - ... other models coming soon (Vicuna etc.)
    - Parameters:
        - Temperature
        - ... more added soon
    - Chat History (Memory):
        - Length (number of messages) of chat to be "remembered" by the AI
- *Vectorstores*
    - Providers:
        - Pinecone
        - ChromaDB
        - ... more added soon (Weaviate, Elasticsearch, ...)
    - Document Retrieval methods:
        - several options here


## Contribution

Contributions to GitChat are welcome! 

These are my current TODOs for the project: [TODO](TODO.txt)

## License

GitChat is licensed under the MIT License. For more information, please see the [LICENSE](LICENSE) file in the repository.
