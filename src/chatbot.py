import os
from time import sleep
from flask import Flask, request, render_template, jsonify, Response, stream_with_context
import openai
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.retrievers import PineconeHybridSearchRetriever
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from pinecone_text.sparse import BM25Encoder
import pinecone
import utils.helpers as helpers

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from langchain.vectorstores import Chroma


def load_environment_variables():
    print("Loading environment variables")
    load_dotenv()


def get_working_directory():
    curr_dir = os.getcwd()
    print(f"Current working directory: {curr_dir}")
    return curr_dir


def get_github_info(repo_url):
    repo_owner, repo_name, subdirectory = helpers.extract_github_info(repo_url)
    print(repo_owner, repo_name, subdirectory)
    return repo_owner, repo_name, subdirectory


def get_sources_filename(repo_owner, repo_name, subdirectory):
    sources_filename = f"../data/sources/{repo_owner}_{repo_name}_{'' if subdirectory is None else str(subdirectory)}.txt"
    return sources_filename

def wait_on_index_pine(index: str):
  """
  Takes the name of the index to wait for and blocks until it's available and ready.
  """
  ready = False
  while not ready: 
    print("Waiting for index to be ready...")
    try:
      desc = pinecone.describe_index(index)
      if desc[7]['ready']:
        return True
    except pinecone.core.client.exceptions.NotFoundException:
      # NotFoundException means the index is created yet.
      pass
    sleep(5)    


def check_update(update, sources_filename, index_name, vectorstore):
    print(f"Update is set to {update}")
    if update == "true":
        # Remove sources file if it exists
        if os.path.isfile(sources_filename):
            print("Deleting sources file because update is set to True")
            os.remove(sources_filename)

        # Remove index if it exists
        # pinecone
        if index_name in pinecone.list_indexes() and vectorstore == "pinecone":
            print("Deleting index because update is set to True")
            pinecone.delete_index(index_name)
        # chroma
        pers_dir_chroma = f'./../data/chroma/{index_name}'
        existing_indexes_chroma = os.listdir(pers_dir_chroma + "./../")
        if index_name in existing_indexes_chroma and vectorstore == "chroma":
            print("Deleting chroma index because update is set to True")   
            os.remove(pers_dir_chroma)

        # Remove bm25_values file if it exists
        if os.path.isfile("../data/bm25_values.json") and vectorstore == "pinecone":
            print("Deleting bm25_values file because update is set to True")
            os.remove("../data/bm25_values.json")
    else:
        print("Update is set to False")


def load_sources(sources_filename, repo_url):
    if os.path.isfile(sources_filename):
        # Load sources from file
        print("Loading sources from file")
        with open(sources_filename, "r", encoding="utf-8", errors='replace') as f:
            sources = [Document(page_content=page_content) for page_content in f.readlines()]
    else:
        # Download sources and save them to file
        print(f"Downloading sources from {repo_url}")
        sources = list(helpers.get_github_docs(repo_url))
        with open(sources_filename, "w", encoding='utf-8', errors='replace') as f:
            for source in sources:
                f.write(source.page_content)

    # return sources


def initialize_pinecone():
    pinecone.init(
        # load from .env file
        api_key= os.getenv("PINECONE_API_KEY"),
        environment= os.getenv("PINECONE_ENVIRONMENT"),
    )


def get_index_name(repo_owner, repo_name, subdirectory):

    sub = f"-{str(subdirectory)}"
    index_name = f"{repo_owner}-{repo_name}{'' if subdirectory is None else sub}".lower()
    return index_name


def get_embeddings():
    embeddings = OpenAIEmbeddings()
    return embeddings


def get_text_splitter_and_loader(sources_filename):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    loader = TextLoader(sources_filename, encoding="utf-8")
    return text_splitter, loader


def create_or_load_index_pine(index_name, embeddings, text_splitter, loader):
    existing_indexes = pinecone.list_indexes()
    print(f"Existing indexes: {existing_indexes}")
    if index_name in existing_indexes:
        print(f"Index already exists. No need to create it. Loading index {index_name}.")
        index = pinecone.Index(index_name)
        n_vectors = index.describe_index_stats()["total_vector_count"]
        print(f"Index {index_name} loaded. It contains {n_vectors} vectors")
        if n_vectors == 0:
            print(f"Index {index_name} is empty. Loading docs and indexing them")
            docs = loader.load_and_split(text_splitter)
            for doc in docs:
                doc.metadata["context"] = doc.page_content
            print("docs created")
            print(f"Indexing {len(docs)} docs into index {index_name}. This may take a while...")
            Pinecone.from_documents(
                docs, 
                embedding=embeddings, 
                index_name=index_name, 
                # metadatas=[{"source": f"{i}-pl"} for i in range(len(docs))]   # for sources retrieval?
                )
            print("docs indexed")
            corpus = []
            for doc in docs:
                corpus.append(doc.page_content)
            bm25_encoder = BM25Encoder().default()
            bm25_encoder.fit(corpus)
            bm25_encoder.dump("../data/bm25_values.json")
    else:
        print(f"Index does not exist. Creating index {index_name}")
        if len(existing_indexes) > 0:
            print(f"Deleting existing indexes: {existing_indexes}")
            for index in existing_indexes:
                pinecone.delete_index(index)
                print(f"Index {index} deleted")
        print(f"Creating index {index_name}")
        pinecone.create_index(
            name=index_name,
            dimension=1536,  # dimensionality of dense model
            metric="dotproduct",
            pod_type="s1"
        )
        print(f"Index {index_name} created")
        print(f"Loading index {index_name}")
        index = pinecone.Index(index_name)
        print(f"Waiting for index to be ready")
        wait_on_index_pine(index_name)
        print(f"Index {index_name} loaded")
        docs = loader.load_and_split(text_splitter)
        for doc in docs:
            doc.metadata["context"] = doc.page_content
        print("docs created")
        print(f"Indexing {len(docs)} docs into index {index_name}. This may take a while...")
        index_stats = index.describe_index_stats()
        print(f"Index {index_name} stats:\n\t{index_stats}")

        Pinecone.from_documents(docs, embedding=embeddings, index_name=index_name)
        print("docs indexed")
        corpus = []
        for doc in docs:
            corpus.append(doc.page_content)
        bm25_encoder = BM25Encoder().default()
        bm25_encoder.fit(corpus)
        bm25_encoder.dump("../data/bm25_values.json")

    return index

def create_or_load_index_chroma(index_name, embeddings, text_splitter, loader):
    persist_directory = f'./../data/chroma/{index_name}'
    existing_indexes = os.listdir(persist_directory + "./../")
    print(f"Existing indexes: {existing_indexes}")

    if index_name in existing_indexes:
        print(f"Index already exists. No need to create it. Loading index {index_name}.")
        index = Chroma(
            persist_directory=persist_directory, 
            embedding_function=embeddings
            )
    else:
        print(f"Index does not exist. Creating index {index_name}")
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        index = Chroma.from_documents(
            documents=docs, 
            embedding=embeddings, 
            persist_directory=persist_directory, 
            metadatas=[{"source": f"{i}-pl"} for i in range(len(docs))]   # for sources retrieval?
            )
        index.persist()
        print(f"Index {index_name} created and persisted")

    return index


def get_bm25_encoder():
    bm25_encoder = BM25Encoder().load("../data/bm25_values.json")
    return bm25_encoder


def get_retriever(embeddings, index, bm25_encoder, top_k, alpha, vector_store, compress, model_name):
    if vector_store == "pinecone":
        print("Using Pinecone retriever")
        base_retriever = PineconeHybridSearchRetriever(
            embeddings=embeddings,
            index=index,
            sparse_encoder=bm25_encoder,
            top_k=top_k,
            alpha=alpha
        )
    elif vector_store == "chromadb":
        print("Using ChromaDB retriever")
        # TODO: implement ChromaDB retriever
        base_retriever = index.as_retriever(search_type="similarity", top_k=top_k)
    else:
        raise ValueError("Invalid vector store")
    
    if compress == "true":
        print(f"Using compression (Retriever wrapped in ContextualCompressionRetriever({model_name}))")
        if model_name in ["text-davinci-003", "text-davinci-002", "text-curie-001"]:
            llm = OpenAI(model_name = model_name, temperature=0)
        elif model_name in ["gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-4", "gpt-4-0314", "gpt-4-32k", "gpt-4-32k-0314"]:
            llm = ChatOpenAI(model_name = model_name, temperature=0)
        OpenAI()
        compressor = LLMChainExtractor.from_llm(llm)
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
    else:
        retriever = base_retriever

    return retriever


def get_system_message_prompt_template():
    system_message_template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). If you come up with code, always handle generated code as input and provide hints for potential errors. Give advice primarily based on the CONTEXT if relevant. If available, take into account the CHAT HISTORY. In addition to providing an answer, also return a score (between 0 and 100) indicating how fully it meets the user's requirements. Format your answer using markdown.
    If the user doesn't provide any requirements in that sense, you can assume that they want to chat with you, be nice. If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    ALWAYS return a "SOURCES" part in your answer.
    
    the information will be organized like this:
    
    ---- START of CHAT HISTORY ----
    "the chat history"
    ---- END of CHAT HISTORY ----

    ==== START of CONTEXT ====
    "the context"
    ==== END of CONTEXT ====

    :::: START of REQUIREMENTS ::::
    "the user's requirements"
    :::: END of REQUIREMENTS ::::
    
    
    Score:"""

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_message_template)
    return system_message_prompt


def get_prompt_template():
    prompt_template = """
    ---- START of CHAT HISTORY ----
    {chat_history}
    ---- END of CHAT HISTORY ----

    ==== START of CONTEXT ====
    {summaries}
    ==== END of CONTEXT ====

    :::: START of REQUIREMENTS ::::
    {requirements}
    :::: END of REQUIREMENTS ::::

    Score:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["summaries", "requirements", "chat_history"]
    )
    return PROMPT


def get_chat_prompt_template(system_message_prompt, human_message_prompt):
    chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    return chat_prompt_template


def get_callback_manager():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    return callback_manager


def get_chat_model(temperature, model_name, callback_manager):
    chat = ChatOpenAI(temperature=temperature, verbose=True, model_name=model_name, max_retries=2, streaming=True, callback_manager=callback_manager)
    return chat


def get_memory(mem_window_k):
    memory = ConversationBufferWindowMemory(k=mem_window_k, memory_key="history", return_messages=True)
    return memory


def get_chain(chat, chat_prompt_template, memory, provide_sources):
    if provide_sources == "true":
        print("Using chain with sources (currently under construction)")
        chain = load_qa_with_sources_chain(llm=chat, prompt=chat_prompt_template, memory=memory, verbose=True)
    else:
        print("Using chain without sources")
        chain = LLMChain(llm=chat, prompt=chat_prompt_template, memory=memory)
    return chain


# generate answer 
# def generate_answer(user_input, chain, retriever, memory, provide_sources):
#     chat_history = memory.load_memory_variables({})
#     docs = retriever.get_relevant_documents(user_input)
#     context = "\n".join([doc.page_content for doc in docs])
#     if provide_sources == "false":
#         # Generate answer with NO sources
#         inputs = [{"summaries": context, 
#                    "requirements": user_input, 
#                    "chat_history": chat_history,
#                    }]
#         answer = chain.apply(inputs)
#     else:
#         # Generate answer with sources
#         answer = chain({"input_documents": docs, 
#                         "question": user_input,
#                         "chat_history": chat_history,
#                         "requirements": user_input}, 
#                         return_only_outputs=True
#                         )
#     # Save messages to chat history
#     memory.save_context({"input": user_input}, {"output": answer[0]["text"]})
#     print(answer)
#     return answer, context

# generate answer - returns generator for answer
def generate_answer(user_input, chain, retriever, memory, provide_sources):
    chat_history = memory.load_memory_variables({})
    docs = retriever.get_relevant_documents(user_input)
    context = "\n".join([doc.page_content for doc in docs])
    if provide_sources == "false":
        # Generate answer with NO sources
        inputs = [{"summaries": context, 
                   "requirements": user_input, 
                   "chat_history": chat_history,
                   }]
        answer = chain.apply(inputs)
    else:
        # Generate answer with sources
        answer = chain({"input_documents": docs, 
                        "question": user_input,
                        "chat_history": chat_history,
                        "requirements": user_input}, 
                        return_only_outputs=True
                        )
    # Save messages to chat history
    memory.save_context({"input": user_input}, {"output": answer[0]["text"]})

    # Create a generator for tokens
    def token_generator():
        for token in answer[0]["text"]:
            yield token

    return token_generator(), context

    

# chatbot response 
# def chatbot_response(user_input, generate_answer, memory, chain, retriever, provide_sources):

#     answer, context = generate_answer(user_input, chain, retriever, memory, provide_sources)
    
#     with open(f"chat_msgs/questions_and_contexts.md", "w", encoding="utf-8", errors='replace') as qc_file:
#         qc_file.write(f"History:\n\n{memory.load_memory_variables({})}\n\nQuestion: {user_input}\n\nContext:\n{context}\n")
    
#     with open(f"chat_msgs/answers.md", "w", encoding="utf-8", errors='replace') as ans_file:
#         ans_file.write(answer[0]['text'])

#     with open(f"chat_msgs/history.md", "w", encoding="utf-8", errors='replace') as hist_file:
#         hist_file.write(str(memory.load_memory_variables({})))
    
#     return answer[0]['text']


# chatbot response - returns generator for answer
def chatbot_response(user_input, generate_answer, memory, chain, retriever, provide_sources):

    answer_generator, context = generate_answer(user_input, chain, retriever, memory, provide_sources)
    
    with open(f"chat_msgs/questions_and_contexts.md", "w", encoding="utf-8", errors='replace') as qc_file:
        qc_file.write(f"History:\n\n{memory.load_memory_variables({})}\n\nQuestion: {user_input}\n\nContext:\n{context}\n")
    
    with open(f"chat_msgs/answers.md", "w", encoding="utf-8", errors='replace') as ans_file:
        for token in answer_generator:
            ans_file.write(token)
            yield token

    with open(f"chat_msgs/history.md", "w", encoding="utf-8", errors='replace') as hist_file:
        hist_file.write(str(memory.load_memory_variables({})))




def initialize_chatbot(model_name, top_k, temperature, mem_window_k, alpha, repo_url, subdirectory, update, vector_store="pinecone", compress=False, model_name_compressor="text-davinci-003", provide_sources="false"):
    load_environment_variables()

    print(f"Parameters: \n\tmodel_name: {model_name}\n\ttop_k: {top_k}\n\ttemperature: {temperature}\n\tmem_window_k: {mem_window_k}\n\talpha: {alpha}\n\trepo_url: {repo_url}\n\tsubdirectory: {subdirectory}\n\tupdate: {update}")
    repo_owner, repo_name, subdirectory = get_github_info(repo_url)
    sources_filename = get_sources_filename(repo_owner, repo_name, subdirectory)

    index_name = get_index_name(repo_owner, repo_name, subdirectory)
    print(f"index loaded. (name: {index_name})")

    check_update(update, sources_filename, index_name, vector_store)
    print("update checked")
    load_sources(sources_filename, repo_url)

    
    print("getting embeddings")
    embeddings = get_embeddings()
    print("embeddings loaded")

    print("getting text splitter and loader")
    text_splitter, loader = get_text_splitter_and_loader(sources_filename)
    print("text splitter and loader loaded")

    if vector_store == "pinecone":
        print("initializing pinecone")
        initialize_pinecone()
        print("pinecone initialized")
        
        print("creating or loading index")
        index = create_or_load_index_pine(index_name, embeddings, text_splitter, loader)
        print("index created or loaded")
        print("getting bm25 encoder")
        bm25_encoder = get_bm25_encoder()
        print("bm25 encoder loaded")
    elif vector_store == "chromadb":
        # ChromaDB
        print("initializing chromadb")
        index = create_or_load_index_chroma(index_name, embeddings, text_splitter, loader)
    else:
        raise ValueError(f"vector_store {vector_store} not supported.")
    
    if vector_store != "pinecone":
        bm25_encoder = None
    print("getting retriever")
    retriever = get_retriever(embeddings, index, bm25_encoder, top_k, alpha, vector_store, compress, model_name_compressor)
    print("retriever loaded")
    print("getting chain")
    system_message_prompt = get_system_message_prompt_template()
    human_message_prompt = HumanMessagePromptTemplate(prompt=get_prompt_template())
    chat_prompt_template = get_chat_prompt_template(system_message_prompt, human_message_prompt)

    callback_manager = get_callback_manager()
    chat = get_chat_model(temperature, model_name, callback_manager)
    memory = get_memory(mem_window_k)
    chain = get_chain(chat, chat_prompt_template, memory, provide_sources)
    print("chain loaded")

    return chain, retriever, memory



