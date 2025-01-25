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
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import csv
import os

EMOTIONS = [
    "Sad, Depressed", "Guilty, Ashamed", "Angry, Irritated, Annoyed, Resentful",
    "Frustrated", "Anxious, Worried, Terrified, Nervous, Panicked",
    "Inferior, Inadequate", "Lonely", "Hopeless, Discouraged", "Happy", "Neutral"
]

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")


class Chatbot:
    def __init__(self):
        # Load and split documents
        loader = TextLoader('stress.txt')
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        self.docs = text_splitter.split_documents(documents)

        # Initialize embeddings with explicit model
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        # Define index name
        self.index_name = "langchain-demo"
        self.PINECONE_ENV = "us-east-1" 
        # Check if the index exists, create if it doesn't
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # Dimension for the MiniLM model
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=self.PINECONE_ENV)
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

        # Initialize HuggingFace InferenceClient
        self.client = InferenceClient(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1", 
            token=HUGGINGFACE_API_KEY
        )

    def ask_with_history(self, question, history):
        """Generate a response while considering the conversation history."""
        # Build the conversation history for the prompt
        if history:
            history_text = "\n".join(
                [f"User: {turn['user']}\nBot: {turn['bot']}" for turn in history]
            )
        else:
            history_text = ""

        # Define the complete prompt with optimized instructions
        prompt = f"""
        You are a professional counselor specializing in stress management.
        Respond empathetically to user concerns and provide actionable advice.

        {f"Conversation History:\n{history_text}" if history_text else ""}
        User: {question}
        Bot:
        """

        # Generate response using the InferenceClient
        response = self.client.text_generation(
            prompt.strip(),
            max_new_tokens=150,
            temperature=0.5,
            top_k=20
        )
        return response.strip()



    def detect_emotion(self, text):
        """Detect emotion based on predefined keywords."""
        emotion_keywords = {
            "Sad, Depressed": ["sad, depressed"],
            "Guilty, Ashamed": ["guilty, ashamed"],
            "Angry, Irritated, Annoyed, Resentful": ["angry, irritated, annoyed, resentful"],
            "Frustrated": ["frustrated"],
            "Anxious, Worried, Terrified, Nervous, Panicked": ["anxious, worried, terrified, nervous, panicked"],
            "Inferior, Inadequate": ["inferior, nadequate"],
            "Lonely": ["lonely"],
            "Hopeless, Discouraged": ["hopeless, discouraged"],
            "Happy": ["happy"],
            "Neutral": ["neutral"]
        }

        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text.lower() for keyword in keywords):
                return emotion
        return "Neutral"  # Default if no keywords match



    def log_mood(self, mood, cause):
        """Log the user's mood and cause of stress into a CSV file."""    
        if mood not in EMOTIONS:
            print(f"Invalid mood: {mood}. Not logged.")
            return

        try:
            with open("mood_logs.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([timestamp, mood, cause])
            print(f"Logged Mood: {mood}, Cause: {cause}")
        except Exception as e:
            print(f"Error logging mood: {e}")


    def save_feedback_with_emotion(self, question, response, feedback):
        """Save user feedback and emotion to a CSV file."""
        emotion = self.detect_emotion(question)
        with open("feedback.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([question, response, feedback, emotion])
        print("Feedback and emotion saved!")

    def generate_pdf_report(self):
        """Generate a PDF report summarizing stress and mood trends."""
        try:
            # Load mood data
            df = pd.read_csv("mood_logs.csv", names=["Timestamp", "Mood", "Cause"])
            mood_counts = df["Mood"].value_counts()
            most_common_cause = df["Cause"].value_counts().idxmax() if not df.empty else "No data"

            # File path for the PDF
            pdf_file = "stress_report.pdf"

            # Generate the mood trends chart
            chart_path = self.plot_mood_trend()  # This generates and saves "mood_trend.png"

            # Create the PDF
            c = canvas.Canvas(pdf_file, pagesize=letter)
            c.setFont("Helvetica", 12)
            
            c.drawString(100, 750, "Personalized Stress Report")
            c.drawString(100, 730, f"Total Mood Entries: {len(df)}")
            y_position = 710


            for mood, count in mood_counts.items():
                c.drawString(100, y_position, f"- {mood}: {count}")
                y_position -= 20

            c.drawString(100, y_position, f"Top Cause of Stress: {most_common_cause}")
            y_position -= 20


            
            # Add the mood trends chart to the PDF
            if chart_path:
                c.drawString(100, y_position, "Mood Trends Over Time:")
                c.drawImage(chart_path, 100, y_position - 250, width=400, height=200)  # Adjust chart size and position
                y_position -= 270  # Adjust space for the chart


            c.drawString(100, 200, "Thank you for using the Stress Management App!")
            c.save()

            return pdf_file

        except Exception as e:
            print(f"Error generating PDF report: {e}")
            return None


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


    def analyze_feedback(self):
        """Analyze feedback distribution and identify problematic responses."""
        try:
            # Load feedback data
            feedback_df = pd.read_csv("feedback.csv", names=["Question", "Response", "Feedback", "Emotion"])

            # Analyze feedback distribution
            feedback_counts = feedback_df["Feedback"].value_counts()

            # Plot feedback distribution
            plt.figure(figsize=(6, 4))
            feedback_counts.plot(kind="bar", title="Feedback Distribution", xlabel="Feedback Type", ylabel="Count")
            plt.tight_layout()
            dist_path = "feedback_distribution.png"
            plt.savefig(dist_path)
            plt.close()

            # Identify problematic responses
            negative_feedback = feedback_df[feedback_df["Feedback"] == "negative"]
            problematic_responses = negative_feedback["Response"].value_counts().head(10)

            return {
                "distribution_chart": dist_path,
                "problematic_responses": problematic_responses
            }
        except Exception as e:
            print(f"Error analyzing feedback: {e}")
            return None


    def correlate_feedback_with_emotions(self):
        """Analyze how feedback correlates with user emotions."""
        try:
            feedback_df = pd.read_csv("feedback.csv", names=["Question", "Response", "Feedback", "Emotion"])
            emotion_feedback = feedback_df.groupby(["Emotion", "Feedback"]).size().unstack(fill_value=0)

            # Plot correlation
            plt.figure(figsize=(8, 6))
            emotion_feedback.plot(kind="bar", stacked=True, title="Feedback Correlation with Emotions", ylabel="Count")
            plt.tight_layout()
            corr_path = "emotion_feedback_correlation.png"
            plt.savefig(corr_path)
            plt.close()

            return corr_path
        except Exception as e:
            print(f"Error analyzing feedback-emotion correlation: {e}")
            return None
