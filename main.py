from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from textblob import TextBlob
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import csv
import os


os.environ["PINECONE_API_KEY"] = "pcsk_77UPqf_PjNe9R441XdWKwahkgcxvA9iXN4cvjX1sjM4EytUQqyVJgYEqa2GqNsZttH8qMM"

# Pinecone API key and environment
PINECONE_API_KEY = "pcsk_77UPqf_PjNe9R441XdWKwahkgcxvA9iXN4cvjX1sjM4EytUQqyVJgYEqa2GqNsZttH8qMM"
PINECONE_ENV = "us-east-1"  # Replace with your environment

# HuggingFace API key
HUGGINGFACE_API_KEY = "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"

class Chatbot:
    def __init__(self):
        # Load and split documents
        loader = TextLoader('stress.txt')
        documents = loader.load()

        #print(f"Loaded Documents:\n{documents}\n")

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        self.docs = text_splitter.split_documents(documents)

        #print(f"Split Documents:\n{self.docs}\n")

        # Initialize embeddings with explicit model
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        # Define index name
        self.index_name = "langchain-demo"

        # Check if the index exists, create if it doesn't
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # Dimension for the MiniLM model
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
            )
            # Populate the index with documents
            self.docsearch = LangChainPinecone.from_documents(
                self.docs, self.embeddings, index_name=self.index_name
            )
        else:
            # Connect to the existing index
            self.docsearch = LangChainPinecone.from_existing_index(
                self.index_name, self.embeddings
            )

        # Initialize HuggingFace LLM
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        #repo_id=endpoint_url="https://api-inference.huggingface.co/models/facebook/opt-125m"
        # self.llm = HuggingFaceHub(
        #     repo_id=repo_id,
        #     model_kwargs={"temperature": 0.8, "top_k": 50},
        #     huggingfacehub_api_token=HUGGINGFACE_API_KEY
        # )
        # self.llm = HuggingFaceEndpoint(
        #     endpoint_url=repo_id, 
        #     huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        #     timeout=300,
        #     temperature=0.8,
        #     top_k=50, 
        #     max_new_tokens=150, 
        #       # Ensure this is <= 250
        # )
       
        # Initialize HuggingFace InferenceClient
        self.client = InferenceClient(
            #model="facebook/opt-350m",  # Use your chosen model
            model="mistralai/Mixtral-8x7B-Instruct-v0.1", 
            token=HUGGINGFACE_API_KEY
        )
        # Define prompt template
        template="""
            You are a professional counselor who provides actionable and empathetic advice to people feeling stressed.
            Below is some context about their situation, followed by their specific question.

            Context:
            {context}

            Question:
            {question}

            Please provide a concise answer:
        """

        self.prompt = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )

    #     #Set up RAG pipeline
    #     self.rag_chain = (
    #         {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}
    #         | self.prompt
    #         | self.llm
    #         | StrOutputParser()
    #     )

    # def ask(self, question):
    #     response = self.rag_chain.invoke(question)
    #     return response.strip()
    def ask_with_history(self, question, history):
        """Generate a response while considering the conversation history."""
        # Format the history into the prompt
        history_text = "\n".join(
            [f"User: {turn['user']}\nBot: {turn['bot']}" for turn in history]
        )
        prompt = f"""
        You are a helpful counselor specializing in stress management.
        Below is the conversation history, followed by the user's new question.

        Conversation History:
        {history_text}

        User: {question}
        Bot:
        """
        # Generate response using InferenceClient or your model
        response = self.client.text_generation(
            prompt,
            max_new_tokens=150,
            temperature=0.5,
            top_k=20
        )
        return response.strip()


    # def ask(self, question):
    #     try:
    #         # Retrieve relevant documents
    #         retriever = self.docsearch.as_retriever()
    #         #context_docs = retriever.get_relevant_documents(question)
    #         context_docs = retriever.invoke(question)

    #         # Combine contexts into a single string
    #         context = "\n".join([doc.page_content for doc in context_docs])
            
    #         if not context.strip():
    #             context = "No specific context is available for this question."

    #         # Format the prompt
    #         prompt = self.prompt.format(context=context, question=question)
    #         #print(f"Prompt Sent to Model:\n{prompt}\n")

    #         # Generate response using InferenceClient
    #         response = self.client.text_generation(
    #             prompt,
    #             max_new_tokens=150,
    #             temperature=0.5, #less randomness
    #             top_k=20, #more focused completions
    #             #stop=["--- Question ---", "--- Answer ---"]
    #         )

    #         answer = response.strip().split("Please provide a concise answer:")[-1].strip()
    #         return answer
    #     except Exception as e:
    #         return f"Error: {str(e)}"




    def detect_emotion(self, text):
        """Detect emotion based on sentiment analysis."""
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity  # Polarity ranges from -1 to 1
        if polarity > 0:
            return "positive"
        elif polarity < 0:
            return "negative"
        else:
            return "neutral"
    def log_mood(self, mood, cause):
        """Log the user's mood and cause of stress into a CSV file."""
        with open("mood_logs.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, mood, cause])
        print(f"Logged Mood: {mood}, Cause: {cause}")

    def save_feedback_with_emotion(self, question, response, feedback):
        """Save user feedback and emotion to a CSV file."""
        emotion = self.detect_emotion(question)
        with open("feedback.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([question, response, feedback, emotion])
        print("Feedback and emotion saved!")

    def generate_report(self):
        """Generate a personalized report from mood logs."""
        try:
            # Load the mood logs
            df = pd.read_csv("mood_logs.csv", names=["Timestamp", "Mood", "Cause"])
            
            # Analyze mood distribution
            mood_counts = df["Mood"].value_counts()
            most_common_cause = df["Cause"].value_counts().idxmax()

            # Create report summary
            report = (
                f"**Mood Summary**:\n"
                f"- Happy: {mood_counts.get('Happy', 0)}\n"
                f"- Neutral: {mood_counts.get('Neutral', 0)}\n"
                f"- Stressed: {mood_counts.get('Stressed', 0)}\n\n"
                f"**Top Cause of Stress**:\n- {most_common_cause}\n"
            )
            return report
        except Exception as e:
            return f"Error generating report: {str(e)}"


    def get_mood_data(self):
        """Load mood data from the CSV file."""
        try:
            # Load the mood logs
            df = pd.read_csv("mood_logs.csv", names=["Timestamp", "Mood", "Cause"])
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])  # Convert to datetime
            df["Date"] = df["Timestamp"].dt.date  # Extract date
            return df
        except Exception as e:
            print(f"Error loading mood data: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on error


    def plot_mood_trend(self):
        """Generate a mood trend chart."""
        df = self.get_mood_data()
        if df.empty:
            return None  # No data to plot

        # Group by date and mood count
        mood_counts = df.groupby(["Date", "Mood"]).size().unstack(fill_value=0)

        # Plot mood trends
        plt.figure(figsize=(10, 6))
        mood_counts.plot(kind="line", marker="o", figsize=(10, 6))
        plt.title("Mood Trends Over Time")
        plt.xlabel("Date")
        plt.ylabel("Mood Count")
        plt.grid()
        plt.legend(title="Mood")
        plt.tight_layout()

        # Save the plot to a file
        plt.savefig("mood_trend.png")
        return "mood_trend.png"  # Return the file path


    def plot_interactive_mood_trend(self):
        """Generate an interactive mood trend chart using Plotly."""
        df = self.get_mood_data()
        if df.empty:
            return None

        # Group by date and mood count
        mood_counts = df.groupby(["Date", "Mood"]).size().reset_index(name="Count")

        # Create Plotly figure
        fig = px.line(
            mood_counts,
            x="Date",
            y="Count",
            color="Mood",
            title="Mood Trends Over Time",
            markers=True,
            labels={"Count": "Mood Count", "Date": "Date"}
        )
        return fig

#Instantiate and run the chatbot
# if __name__ == "__main__":
#     bot = Chatbot()
#     user_input = input("Ask me anything about stress: ")
#     response = bot.ask(user_input)
#     print(response)
