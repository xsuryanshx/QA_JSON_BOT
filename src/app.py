from fastapi import FastAPI, File, UploadFile, HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import JSONLoader
import json
import uvicorn
import os
from datetime import datetime
from rag_qa_model import RAG_QA_Model

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Retrieve the OpenAI API key
OPEN_API_KEY = os.getenv("OPENAI_API_KEY")

model = RAG_QA_Model()

def loader_for_context(path_to_file):
    print(path_to_file)
    if path_to_file.split(".")[-1] == "json":
        loader = JSONLoader(file_path = path_to_file,
                jq_schema='.',
                text_content=False)
        print("json file loaded")
    elif path_to_file.split(".")[-1] == "pdf":
        loader = PyPDFLoader(path_to_file)
        print("pdf file loaded")
    else:
        raise Exception("Please select valid input file format eg. pdf or json")

    return loader

def questions_from_json(file_path):
    with open(file_path, "rb") as f:
        questions = json.load(f)
    list_questions = [item["question"] for item in questions]
    return list_questions

# Define endpoint to handle questions and documents
@app.post("/")
async def answer_questions(questions_file: UploadFile = File(...), context_file: UploadFile = File(...)):
    if model.is_valid_api_key(api_key=OPEN_API_KEY):
        model.set_api_key(api_key= OPEN_API_KEY)
        # Save the uploaded file
        file_path = f"./files/{questions_file.filename}"
        with open(file_path, "wb") as f:
            f.write(questions_file.file.read())

        # Determine file type and read questions
        if questions_file.filename.endswith('.json'):
            questions = questions_from_json(file_path)
        else:
            return {"error": "Unsupported file format, input file should be in json format"}

        # Adding context file
        context_file_path = f"./files/{context_file.filename}"
        with open(context_file_path, "wb") as f:
            f.write(context_file.file.read())

        # Determine file type and read questions
        if context_file.filename.endswith('.json'):
            loader = JSONLoader(file_path = context_file_path,
                    jq_schema='.',
                    text_content=False)
        elif context_file.filename.endswith('.pdf'):
            loader = PyPDFLoader(context_file_path)
        else:
            return {"error": "Input file should be in json or pdf format"}

        # Loads the input context data and creates chunks.
        model.load_document(loader)

        # Iterate through questions and generate answers
        answers = []
        for index, question in enumerate(questions):
            answer = model.answer_questions(question, number_of_documents_to_review=5, temperature=0)
            answer = {
                "question": question,
                "answer": answer.replace("\n", ""),
            }
            answers.append(answer)

        with open("./output/answers.json", "w") as final:
            json.dump(answers, final, indent=4)

        return answers
    else:
        raise Exception("Please use correct OpenAI key.")

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=600)