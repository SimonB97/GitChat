import os
import shutil
from time import sleep
import time
import deeplake
# from flask import Flask, request, render_template, jsonify, Response, stream_with_context
# import openai
from dotenv import load_dotenv

from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate
from langchain.vectorstores import Pinecone, DeepLake
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.retrievers import PineconeHybridSearchRetriever
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from pinecone_text.sparse import BM25Encoder
import pinecone
import utils.helpers as helpers

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter
from langchain.vectorstores import Chroma
from deeplake.util.exceptions import ResourceNotFoundException


def load_environment_variables():
    print("Loading environment variables")
    load_dotenv()


def get_working_directory():
    curr_dir = os.getcwd()
    print(f"Current working directory: {curr_dir}")
    return curr_dir


def get_github_info(repo_url):
    repo_owner, repo_name, subdirectory = helpers.extract_github_info(repo_url)
    print(f"DEBUG get_github_info: {repo_owner}, {repo_name}, {subdirectory}")
    return repo_owner, repo_name, subdirectory


# def get_sources_filename(repo_owner, repo_name, subdirectory):
#     subdirectory = subdirectory.replace("/", "_")
#     sources_filename = f"../data/sources/{repo_owner}_{repo_name}_{'' if subdirectory is None else str(subdirectory)}.txt"
#     print(f"DEBUG get_sources_filename: {sources_filename}")
#     return sources_filename

def get_sources_filename(repo_owner, repo_name, subdirectory):
    if subdirectory is not None:
        subdirectory = subdirectory.replace("/", "_")
        if subdirectory.startswith("tree_"):
            subdirectory = subdirectory[5:]
    sources_filename = f"../data/sources/{repo_owner}_{repo_name}{'' if subdirectory is None else f'_{str(subdirectory)}'}.txt"
    print(f"DEBUG get_sources_filename: {sources_filename}")
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
      print("Index not found yet. Waiting...")
      # NotFoundException means the index is created yet.
      pass
    sleep(5)    


def check_update(update, sources_filename, index_name, vectorstore):
    print(f"Update is set to {update}\n Vectorstore is set to {vectorstore}")
    if update == "true":
        # Remove sources file if it exists
        if os.path.isfile(sources_filename):
            print("Deleting sources file because update is set to True")
            os.remove(sources_filename)

        # Remove index if it exists
        
        # pinecone
        # print(pinecone.list_indexes())  # TODO: this function is currently not workin (see github issue), delete index manually for now
        # if index_name in pinecone.list_indexes() and vectorstore == "pinecone":
        #     print("Deleting index because update is set to True")
        #     pinecone.delete_index(index_name)

        # chroma
        pers_dir_chroma = f'./../data/chroma/{index_name}'
        existing_indexes_chroma = os.listdir('./../data/chroma/')
        print(f"Existing chroma indexes: {existing_indexes_chroma}")
        if f"'{index_name}'" in existing_indexes_chroma and vectorstore == "chroma":
            print("Deleting chroma index because update is set to True")   
            shutil.rmtree(pers_dir_chroma)
            print(f"Remaining chroma indexes: {os.listdir(pers_dir_chroma)}")


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
                f.write(source.page_content + "\n\n")

    return sources


def initialize_pinecone():
    pinecone.init(
        # load from .env file
        api_key= os.getenv("PINECONE_API_KEY"),
        environment= os.getenv("PINECONE_ENVIRONMENT"),
    )


def get_index_name(repo_owner, repo_name, subdirectory):
    if subdirectory is not None:
        subdirectory = subdirectory.replace("/", "-")
    sub = f"-{str(subdirectory)}"
    index_name = f"{repo_owner}-{repo_name}{'' if subdirectory is None else sub}".lower()
    print(f"Index name to use: {index_name}")
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


def create_or_load_index(index_name, embeddings, text_splitter, loader, vectorstore):
    # IF PINECONE
    print(f"DEBUG Embeddings create_or_load_index:\n\ttype: {type(embeddings)}\nobject: {embeddings}")
    if vectorstore == "pinecone":
        # existing_indexes = pinecone.list_indexes()   # currently not working
        existing_indexes = []
        print(f"Existing indexes: {existing_indexes}")
        if index_name in existing_indexes:
            print(f"Index already exists. No need to create it. Loading index {index_name}.")
            index = pinecone.Index(index_name)
            wait_on_index_pine(index_name)
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
            print(f"Index does not exist. Creating index {vectorstore} {index_name}")
            if len(existing_indexes) > 0:
                print(f"Deleting existing indexes: {existing_indexes}")
                for index in existing_indexes:
                    pinecone.delete_index(index)
                    print(f"Index {index} deleted")
            print(f"Creating index {index_name}")
            pinecone.create_index(
                name=str(index_name),
                dimension=1536,  # dimensionality of dense model
                metric="dotproduct",
                pod_type="s1"
            )
            print(f"Index {index_name} created")
            print(f"Loading index {index_name}")
            index = pinecone.Index(index_name)
            wait_on_index_pine(index_name)
            print(f"Index {index_name} loaded")
            docs = loader.load_and_split(text_splitter)
            for doc in docs:
                doc.metadata["context"] = doc.page_content
            corpus = []
            for doc in docs:
                corpus.append(doc.page_content)
            bm25_encoder = BM25Encoder().default()
            bm25_encoder.fit(corpus)
            bm25_encoder.dump("../data/bm25_values.json")
            print("docs created")
            print(f"Indexing {len(docs)} docs into index {vectorstore} {index_name}. This may take a while...")
            # index_stats = index.describe_index_stats()   # currently not working
            # print(f"Index {index_name} stats:\n\t{index_stats}")

            Pinecone.from_documents(docs, embedding=embeddings, index_name=index_name)
            print("docs indexed")
    
    # IF CHROMA
    elif vectorstore == "chromadb":
        persist_directory = f'./../data/chroma/{index_name}'
        existing_indexes = os.listdir(persist_directory + "./../")
        print(f"Existing indexes: {existing_indexes}")

        if index_name in existing_indexes:
            print(f"Index already exists. No need to create it. Loading index {vectorstore} {index_name}.")
            index = Chroma(
                persist_directory=persist_directory, 
                embedding_function=embeddings
                )
        else:
            print(f"Index does not exist. Creating index {vectorstore} {index_name}")
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

    # IF DEEPLAKE
    elif vectorstore == "deeplake":
        username = os.getenv("ACTIVELOOP_USER_NAME") # replace with your username from app.activeloop.ai

        max_retries = 120  # 10 minutes with 5 seconds interval
        retry_count = 0

        while retry_count < max_retries:
            try:
                if deeplake.exists(f"hub://{username}/{index_name}"):
                    print(f"Index already exists. No need to create it. Loading index {vectorstore} {index_name}.")
                    index = DeepLake(dataset_path=f"hub://{username}/{index_name}", embedding_function=embeddings, read_only=True, token=os.getenv("ACTIVELOOP_TOKEN"))
                    break
                else:
                    print(f"Index does not exist. Creating index {vectorstore} {index_name}")
                    documents = loader.load()
                    docs = text_splitter.split_documents(documents)
                    index = DeepLake(dataset_path=f"hub://{username}/{index_name}", embedding_function=embeddings, public=False, token=os.getenv("ACTIVELOOP_TOKEN"))
                    index.add_documents(docs)
                    break
            except ResourceNotFoundException as e:
                if retry_count == max_retries - 1:
                    raise Exception("Max retries reached. Error: " + str(e))
                else:
                    print(f"Waiting for deletion of old index. Retrying in 5 seconds...")
                    time.sleep(5)
                    retry_count += 1
            except Exception as e:
                raise e  # Re-raise the exception if it's not a ResourceNotFoundException


    return index




def get_bm25_encoder():
    bm25_encoder = BM25Encoder().load("../data/bm25_values.json")
    return bm25_encoder


def get_retriever(embeddings, index, bm25_encoder, top_k, alpha, vector_store, compress, model_name):
    if vector_store == "pinecone":
        print("Using Pinecone hybrid retriever")
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
        search_type = "mmr"   # "mmr" or "similarity"
        base_retriever = index.as_retriever(search_type=search_type, search_kwargs={"k": int(top_k)})
    elif vector_store == "deeplake":
        base_retriever = index.as_retriever()
        base_retriever.search_kwargs['distance_metric'] = 'cos'
        base_retriever.search_kwargs['fetch_k'] = 100
        base_retriever.search_kwargs['maximal_marginal_relevance'] = True
        base_retriever.search_kwargs['k'] = int(top_k)

        # TODO: optionally add deeplake filter
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
        # compressor = LLMChainFilter.from_llm(llm)
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
    else:
        retriever = base_retriever

    return retriever


def get_system_message_prompt_template():
    system_message_template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). If you come up with code, always handle generated code as input and provide hints for potential errors. Give advice primarily based on the CONTEXT if relevant. If available, take into account the CHAT HISTORY. In addition to providing an answer, also return a score (between 0 and 100) indicating how fully it meets the user's question. Format your answer using markdown.
    If the user doesn't provide any question in that sense, you can assume that they want to chat with you, be nice. If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    ALWAYS return a "SOURCES" part in your answer.
    
    the information will be organized like this:
    
    ---- START of CHAT HISTORY ----
    "the chat chat_history"
    ---- END of CHAT chat_history ----

    ==== START of CONTEXT ====
    "the context"
    ==== END of CONTEXT ====

    :::: START of REQUIREMENTS ::::
    "the user's question"
    :::: END of REQUIREMENTS ::::
    
    
    Score:"""

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_message_template)
    return system_message_prompt


def get_prompt_template():
    prompt_template = """
    ---- START of CHAT chat_history ----
    {chat_history}
    ---- END of CHAT chat_history ----

    ==== START of CONTEXT ====
    {summaries}
    ==== END of CONTEXT ====

    :::: START of REQUIREMENTS ::::
    {question}
    :::: END of REQUIREMENTS ::::

    Score:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["summaries", "question", "chat_history"]
    )
    return PROMPT


def get_chat_prompt_template(system_message_prompt, human_message_prompt):
    chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    return chat_prompt_template


def get_chat_model(temperature, model_name):
    chat = ChatOpenAI(temperature=temperature, verbose=False, model_name=model_name, max_retries=4, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
    return chat


def get_memory(mem_window_k):
    memory = ConversationBufferWindowMemory(k=mem_window_k, memory_key="chat_history", return_messages=True, output_key='answer')
    return memory


def get_chain(chat, chat_prompt_template, memory, provide_sources, chain_type, retriever):
    if provide_sources == "true":
        print("Using chain with sources (currently under construction)")
        qa_chain = load_qa_with_sources_chain(llm=chat, prompt=chat_prompt_template, memory=memory, verbose=False)
        retr_chain = RetrievalQAWithSourcesChain(combine_documents_chain=qa_chain, retriever=retriever,
                                     reduce_k_below_max_tokens=True, max_tokens_limit=3375,
                                     return_source_documents=True)
        tools = [
            Tool(
                name = "Search",
                func=retr_chain,
                description="useful to answer factual questions"
            ),
        ]
        llm=ChatOpenAI(temperature=0)
        chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=False, memory=memory)
    elif provide_sources == "false" and chain_type == "llm_chain":
        print("Using chain without sources (LLMChain)")
        chain = LLMChain(llm=chat, prompt=chat_prompt_template, memory=memory)
    elif chain_type == "conv_retr_chain":
        if provide_sources == "true":
            return_source_documents = True
        else:
            return_source_documents = False
        print("Using Conversational Retrieval Chain")
        chain = ConversationalRetrievalChain.from_llm(
            llm=chat, 
            retriever=retriever, 
            # qa_prompt=chat_prompt_template, 
            memory=memory,
            return_source_documents=return_source_documents,
            verbose=False
            )

    return chain


# generate answer - returns generator for answer
def generate_answer(user_input, chain, retriever, memory, provide_sources, chain_type):
    if chain_type != "conv_retr_chain":
        chat_history = memory.load_memory_variables({})
        print("DEBUG: chat_history\n\t", chat_history)
        docs = retriever.get_relevant_documents(user_input)
        context = "\n".join([doc.page_content for doc in docs])

    if provide_sources == "false" and chain_type == "llm_chain":
        # Generate answer with NO sources
        inputs = [{"summaries": context, 
                   "question": user_input, 
                   "chat_history": chat_history,
                   }]
        answer = chain.apply(inputs)
    elif chain_type == "conv_retr_chain":
        print("DEBUG: Retrieval Chain")
        # chat_history = memory
        chat_history = memory.load_memory_variables({})
        # Generate answer Conversational Retrieval Chain
        answer = chain({
            "question": user_input, 
            "chat_history": chat_history
            })
        if provide_sources == "true":
            context = answer['source_documents']
        else:
            context = ""
    else:
        # Generate answer with sources
        answer = chain.run(user_input)

    # Save messages to chat chat_history
    if chain_type != "conv_retr_chain":
        # Create a generator for tokens
        def token_generator():
            for token in answer[0]["text"]:
                yield token
        if provide_sources == "false":
            memory.save_context({"input": user_input}, {"output": answer[0]["text"]})
        else:
            memory.save_context({"input": user_input}, {"output": answer['answer']})

    else:
        # Create a generator for tokens
        def token_generator():
            for token in answer['answer']:
                yield token
    
    return token_generator(), context


# chatbot response - returns generator for answer
def chatbot_response(user_input, generate_answer, memory, chain, retriever, provide_sources, chain_type):

    answer_generator, context = generate_answer(user_input, chain, retriever, memory, provide_sources, chain_type)

    with open(f"chat_msgs/questions_and_contexts.md", "w", encoding="utf-8", errors='replace') as qc_file:
        qc_file.write(f"History:\n\n{memory.load_memory_variables({})}\n\nQuestion: {user_input}\n\nContext:\n{['source' + str(i) + ': ' + doc.page_content for i, doc in enumerate(context)]}")

    with open(f"chat_msgs/answers.md", "w", encoding="utf-8", errors='replace') as ans_file:
        for token in answer_generator:
            ans_file.write(token)
            yield token.encode('utf-8')  # Convert string to bytes

    with open(f"chat_msgs/chat_history.md", "w", encoding="utf-8", errors='replace') as hist_file:
        hist_file.write(str(memory.load_memory_variables({})))

    if chain_type == "conv_retr_chain":
        yield "\n\n---\n###[Sources]\n\n".encode('utf-8')
        # list of sources (context) to generator
        for i, doc in enumerate(context):
            yield f'\n\n####Source {i + 1}\n\n'.encode('utf-8')
            for token in doc.page_content:
                yield f'{token}'.encode('utf-8')




def initialize_chatbot(model_name, top_k, temperature, mem_window_k, alpha, repo_url, subdirectory, update, chain_type, vector_store="pinecone", compress=False, model_name_compressor="text-davinci-003", provide_sources="false"):
    load_environment_variables()

    print(f"Parameters: \n\tmodel_name: {model_name}\n\ttop_k: {top_k}\n\ttemperature: {temperature}\n\tmem_window_k: {mem_window_k}\n\talpha: {alpha}\n\trepo_url: {repo_url}\n\tsubdirectory: {subdirectory}\n\tupdate: {update}")

    repo_url = repo_url  # + '/' + subdirectory
    repo_owner, repo_name, subdirectory = get_github_info(repo_url)
    sources_filename = get_sources_filename(repo_owner, repo_name, subdirectory)

    index_name = get_index_name(repo_owner, repo_name, subdirectory)

    check_update(update, sources_filename, index_name, vector_store)
    git_sources_paths = load_sources(sources_filename, repo_url)

    embeddings = get_embeddings()
    text_splitter, loader = get_text_splitter_and_loader(sources_filename)
        
    index = create_or_load_index(index_name, embeddings, text_splitter, loader, vector_store)
    if vector_store == "pinecone":
        print("initializing pinecone")
        initialize_pinecone()
        bm25_encoder = get_bm25_encoder()
    else:
        bm25_encoder = None
    
    retriever = get_retriever(embeddings, index, bm25_encoder, top_k, alpha, vector_store, compress, model_name_compressor)
    system_message_prompt = get_system_message_prompt_template()
    human_message_prompt = HumanMessagePromptTemplate(prompt=get_prompt_template())
    chat_prompt_template = get_chat_prompt_template(system_message_prompt, human_message_prompt)

    chat = get_chat_model(temperature, model_name)
    memory = get_memory(mem_window_k)
    chain = get_chain(chat, chat_prompt_template, memory, provide_sources, chain_type, retriever)

    return chain, retriever, memory



