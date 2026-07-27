from dotenv import load_dotenv
import sys

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.output_parsers import StrOutputParser
from pypdf import PdfReader


# Load environment variables from .env file
# There must be an Google api key named GOOGLE_API_KEY in the .env file.
load_dotenv()


def load_pdf(file_path: str) -> list[Document]:
    """Create Documents objects for each page in the PDF file.

    Parameters
    ----------
    file_path : str
        Path to the PDF file.

    Returns
    -------
    List[Document]
        A list of Document objects, one for each page in the PDF.
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

    Parameters
    ----------
    documents : list[Document]
        List of Document objects to be chunked.
    chunk_size : int
        The maximum size of each chunk in characters.
    chunk_overlap: int
        The number of characters to overlap between chunks.

    Returns
    -------
    list[Document]
        A list of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)


def initialize_vectorstore(documents: list[Document]) -> InMemoryVectorStore:
    """Initialize the vector store with embeddings from the PDF document.

    Parameters
    ----------
    documents: list[Document]
        List of Document objects to be added to the vector store.

    Returns
    -------
    InMemoryVectorStore
        An instance of InMemoryVectorStore containing the document embeddings.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-mpnet-base-v2',
        encode_kwargs={'normalize_embeddings': True},
    )
    vectorstore = InMemoryVectorStore(embedding=embeddings)
    vectorstore.add_documents(documents)
    return vectorstore


def join_docs(docs: list[Document]) -> str:
    """Join the content of multiple Document objects into a single string.

    Parameters
    ----------
    docs : list[Document]
        List of Document objects to be joined.

    Returns
    -------
    str
        A single string containing the concatenated content of all
        documents.
    """
    return '\n\n'.join(doc.page_content for doc in docs)


# The template to 
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    (
        'system',
        'You are a helpful FAQ chatbot that answers questions based on the provided manual. If the answer is not contained within the documents, respond with "I don\'t know."'
    ),
    (
        'human',
        """Answer the question based only on the following context:

        {context}


        Question:
        
        {question}

        Provide a detailed answer:
        """
    )
])


def main():
    """Program runner"""
    # Setting up data
    print('Loading PDF...')
    try:
        pages = load_pdf('./resources/fridge manual.pdf')
    except FileNotFoundError:
        print('Could not find the data file.', '\n', 'Exiting program.')
        sys.exit()
    print(f'Loaded PDF with {len(pages)} pages.')
    # Break up pages into smaller chunks
    chunked_pages = chunk_documents(pages)
    print(f'Chunked PDF into {len(chunked_pages)} chunks.')
    print('Initializing vector store...')
    retriever = (
        initialize_vectorstore(chunked_pages)
        .as_retriever(search_kwargs={'k': 3})
    )
    print('Vector store initialized.')

    print('Initializing agent...')
    llm = ChatGoogleGenerativeAI(
        model='gemini-3.6-flash',
        streaming=True,
    )
    print('Agent initialized.')

    # Main program loop
    while True:
        user_input = input('Type your question or type "q" to exit: ')

        # If input was empty, restart loop
        if not user_input:
            continue

        if (user_input.lower() == 'q'):
            print('Exiting...')
            break

        print('Searchin vector store for relevant information...')
        docs_string = join_docs(retriever.invoke(user_input))
        context = {'context': docs_string, 'question': user_input}
        parser = StrOutputParser()

        # Create langchain pipeline
        chain = PROMPT_TEMPLATE | llm | parser

        
        # Print generated text to the terminal as it's generated
        for chunk in chain.stream(context):
            print(chunk, end='', flush=True)

        # Print new line after content
        print('\n')
        user_input = input('Press "Enter" to continue or "q" to quit: ')
        if user_input.lower() == 'q':
            break


if __name__ == '__main__':
    main()
