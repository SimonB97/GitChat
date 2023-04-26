<h1 align="center">GitChat</h1>


GitChat is a chatbot that helps users to get insights into GitHub repositories. It is designed to provide interactive assistance and support for users working with files in a GitHub repository. By integrating chat capabilities, the chatbot can help users  understand code and answer questions related to the project. The chatbot utilizes OpenAI LLMs, Langchain and Pinecone to provide answers to questions about the repository.

## Disclaimer

This project is still in development and is not ready for production use.
Also, because the results are generated by Large Language Models, which are probabilistic, the results may not always be correct.
## Installation

To install GitChat, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies using: \
    `pip install -r requirements.txt`
3. Navigate to the `src` directory.
4. Run the chatbot using `python app.py`.

## Prerequisites

- copy `.env.example` to `.env` and fill in the required fields.
    - only the keys to providers being used are required

## Usage

To use GitChat, follow these steps:

1. **Start the application** by running `python app.py`.
2. This will open a web browser and **navigate to the start page** (http://\<your-ip-address>:5000/start - your-ip-address is the ip address of the machine running the application, will be automatically detected)
3. **Fill in the required fields** on the start page and click **Submit** to initialize the chatbot. 
 (Note: Depending on the size of the repository and the Vectorestore used, this may take a few minutes.)
4. Once the chatbot is initialized, you will be redirected to the **main chat interface**.
5. Send Messages using **Send** button or **Enter** key.
6. Clear the Chat History using the **Clear** button.

### Tip: Most promising configurations:
*Feel free to try out other combinations, but these are the ones that I found to work best so far:*
*(Please share your best working configurations in the issues section!)*

- **LLM**: GPT-4
- **Vectorstore**: Pinecone
- **Document Retrieval**: HybridSearch
- **Chat History**: ~5 messages
- **Temperature**: ~0.2
- **Top K**: 30
- **Alpha**: 0.5
- **Compression**: enabled
- **Compression model**: GPT-3.5-Turbo

## Features

- Question Answering powered by OpenAI LLMs (and others to come)
- Chat History (AI has short term memory, like ChatGPT)
- Text sections of files relevant to the question are passed to the AI as context
    - Questions are answered based on the relevant context
- Easily Customizable Options for LLM, Vectorstore and retrieval method

### Configuration options

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
        - HybridSearch (currently only available for Pinecone)
        - several options here

> Although I will likely add a bunch of options in the future, (obviously) I do not have the time to add all of the ones available in LangChain. So this won't be some kind of "sandbox".  Nevertheless, I will still try to make it as customizable as possible (focusing on the most promising, latest features offered by LangChain regarding Vectorstores, LLMs and Document Retrieval), as Document Retrieval use cases vary a lot and I want to make it as easy as possible to try out different combinations for the best results.


## Contribution

Contributions to GitChat are welcome! 

These are my current TODOs for the project: [TODO](TODO.txt)

## License

GitChat is licensed under the MIT License. For more information, please see the [LICENSE](LICENSE) file in the repository.
