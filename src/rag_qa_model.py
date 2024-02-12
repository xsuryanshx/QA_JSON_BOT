from typing import Tuple
import openai
import pandas as pd
import time
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.callbacks import get_openai_callback


class RAG_QA_Model:
    """QA Model Class"""

    def __init__(self) -> None:
        """initializer"""
        _ = load_dotenv(find_dotenv())
        self.__db = None
        self.__documents = []

    def load_document(
        self,
        loader
    ):
        """load documents

        Args:
            loader : PDF/JSON Loader from input context file.
        """
        self.__documents = loader.load()

        # split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=100
        )
        self.__documents = text_splitter.split_documents(self.__documents)

        # Use the Open AI Embeddings
        embeddings = OpenAIEmbeddings()
        self.__db = Chroma.from_documents(self.__documents, embeddings)

    def is_valid_api_key(self, api_key: str) -> bool:
        """
        Determine whether the input api key is valid.

        Parameters
        ----------
        api_key: str
            An API key

        Returns
        -------
        api_key_is_valid: bool
            Whether the API key is valid or not
        """
        try:
            test = OpenAI(openai_api_key=api_key, max_tokens=2)
            test("test")
        except openai.error.AuthenticationError:
            return False
        else:
            return True

    def set_api_key(self, api_key: str) -> None:
        """
        Set the api key.

        Parameters
        ----------
        api_key: str
            An API key
        """
        openai.api_key = api_key

    def answer_questions(
        self,
        question: str,
        number_of_documents_to_review: int,
        temperature: float,
    ) -> Tuple[pd.DataFrame, float]:
        """model used to answer question based on input question and parameters

        Args:
            question (str): question in string
            number_of_documents_to_review (int): number of most chunks of text used to answer
            temperature (float): temperature

        Returns:
            Tuple[pd.DataFrame, float]: dataframe of answer
        """
        retriever = self.__db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": number_of_documents_to_review},
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=temperature),
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt_template},
        )
        start_time = time.time()

        similar_documents = retriever.get_relevant_documents(question)

        with get_openai_callback() as cb:
            result = qa_chain({"input_documents": similar_documents, "query": question})

        end_time = time.time()
        total_request_time = round(end_time - start_time)

        return result["result"]

    @property
    def prompt_template(self):
        """Prompt for generating answer."""
        template_format = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        ALways remember to give only the answer of questions who's data you can find in the "Context:". 
        If you can't find the relevant information in "Context:" please return "Sorry, I don't know about it"

        Examples of some expected answers - 

        Examples #1
        Context: The witness took the stand as directed. It was night and the witness forgot his glasses. \
        he was not sure if it was a sports car or an suv. The rest of the report shows everything was okay.

        Question: what type was the car?
        Answer: He was not sure if it was a sports car or an suv.

        Examples #2
        Context: Pears are either red or orange

        Question: What are your network security protocols?
        Answer: Sorry, I don't know about it.

        Now your turn, Begin!

        Summary: {context}
        Question :{question}
        Answer:
        """
        prompt = PromptTemplate(
            template=template_format,
            input_variables=["context", "question"],
        )
        return prompt

