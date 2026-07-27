from dotenv import load_dotenv
from os import getenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from pypdf import PdfReader


load_dotenv()

OPENAPI_KEY = getenv("OPENAI_API_KEY")
GENAI_API_KEY = getenv("GENAI_API_KEY")

def load_pdf(file_path: str) -> list[Document]:
    """Create Documents objects for each page in the PDF file.

    Parameters:
        file_path (str): Path to the PDF file.

    Returns:
        List[Document]: A list of Document objects, one for each page in the PDF.
    """
    pdf_reader = PdfReader(file_path)
    documents = [
        Document(
            page_content=page.extract_text(),
            metadata={'source': file_path, 'page': i + 1}
        )
        for i, page in enumerate(pdf_reader.pages)
    ]
    return documents


def chunk_documents(
        documents: list[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> list[Document]:
    """Chunk the documents into smaller pieces.

    Parameters:
        documents (list[Document]): List of Document objects to be chunked.
        chunk_size (int): The maximum size of each chunk in characters.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        list[Document]: A list of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)


def initialize_vectorstore(
    documents: list[Document],
) -> InMemoryVectorStore:
    """Initialize the vector store with embeddings from the PDF document.

    Parameters:
        documents (list[Document]): List of Document objects to be added to the vector store.

    Returns:
        InMemoryVectorStore: An instance of InMemoryVectorStore containing the document embeddings.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-mpnet-base-v2',
        encode_kwargs={'normalize_embeddings': True},
    )
    vectorstore = InMemoryVectorStore(embedding=embeddings)
    vectorstore.add_documents(documents)
    return vectorstore


def query_vectorstore(
    query: str,
    vectorstore: InMemoryVectorStore,
)-> list[Document]:
    """Query the vector store for relevant documents.

    Parameters:
        query (str): The query string to search for.
        vectorstore (InMemoryVectorStore): The vector store to query.

    Returns:
        list[Document]: A list of Document objects that are relevant to the query.
    """
    return vectorstore.similarity_search(query, k=3)

def join_docs(docs: list[Document]) -> str:
    """Join the content of multiple Document objects into a single string.

    Parameters:
        docs (list[Document]): List of Document objects to be joined.

    Returns:
        str: A single string containing the concatenated content of all documents.
    """
    return '\n\n'.join(doc.page_content for doc in docs)

prompt_template = ChatPromptTemplate.from_messages([
    (
        'system',
        "You are a helpful assistant that answers questions based on the provided documents. If the answer is not contained within the documents, respond with \"I don't know.\""
    ),
    (
        'human',
        """Answer the question base only on the following context:

        {context}


        Question: {question}

        Provide a detailed answer:
        """
    )
])
def main():
    print('Loading PDF...')
    pages = load_pdf('./resources/fridge manual.pdf')
    print(f'Loaded PDF with {len(pages)} pages.')
    chunked_pages = chunk_documents(pages)
    print(f'Chunked PDF into {len(chunked_pages)} chunks.')
    print('Initializing vector store...')
    retriever = initialize_vectorstore(chunked_pages).as_retriever(search_kwargs={'k': 3})
    print('Vector store initialized.')
    print('Initializing agent...')

    llm = ChatGoogleGenerativeAI(
        model='gemini-3.6-flash',
        streaming=True,
    )
    print('Agent initialized.')
    while True:
        user_input = ''
        user_input = input('Type your query or type "q" to exit: ')
        if not user_input:
            continue
        if (user_input.lower() == 'q'):
            print('Exiting...')
            break
        print('Getting relevant documents from vector store...')
        context = join_docs(retriever.invoke(user_input))
        prompt = prompt_template.format_prompt(context=context, question=user_input)
        chain = prompt | llm
        for chunk in chain.stream():
            print(chunk, end='', flush=True)
        # print(f"Answer: {response.content}")


if __name__ == '__main__':
    main()
