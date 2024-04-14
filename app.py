import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from create_database import CreateDatabase

st.set_page_config(page_title="PaperPundit", layout="wide")

st.markdown("""
## ChatDocs: Def Not for Medical Emergencies, Just Document Emergencies

Warning: My powers may seem magical, but trust me, I'm no genie. Use your brain before taking my word as law. You know, just in case I decide to sprinkle a little mischief instead of wisdom.

### How It Works

Follow these simple steps to interact with the chatbot:
1. **Enter Your API Key**: You'll need a Google API key for the chatbot to access Google's Generative AI models. Obtain your API key https://makersuite.google.com/app/apikey.

2. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

3. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")


api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

def get_conversational_chain():
    prompt_template = """
    Answer the question from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the Document", don't provide the wrong answer\n\n
    
    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Search the DB.
    context_text = db.similarity_search(user_question, k=3)
    
    print("---------------Context Text------------------------------")
    print(context_text)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": context_text, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    add_vertical_space(2)
    user_question = st.text_input("Ask a Question from the PDF not from your thrilling personal saga", key="user_question")

    if user_question and api_key:  # Ensure API key and user question are provided
        user_input(user_question)

    with st.sidebar:
        st.title("💬 PDFPundit:")
        pdf_docs = st.file_uploader("Just upload your PDF and hit me hard", accept_multiple_files=True, key="pdf_uploader")
        if st.button("HIT ME", key="process_button"):
            with st.spinner("Magic in Progress..."):
                vectorDb = CreateDatabase(api_key)
                vectorDb.create_vector_database(pdf_docs)
                st.success("Done", icon='✅')
        
        add_vertical_space(5)
        st.write('Made with ❤️ and a touch of Copy-Paste magic by Rahul Pandey')

if __name__ == "__main__":
    main()