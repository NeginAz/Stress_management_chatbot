# Stress Management Chatbot

This repository contains a Stress Management Chatbot application designed to provide users with empathetic and actionable advice for managing stress. The chatbot leverages AI, Retrieval-Augmented Generation (RAG), and user feedback to continuously improve its responses and user experience.


## Features

### 1. Chatbot with Multi-Turn Conversations

- Maintains context across multiple user queries.

- Provides concise and empathetic advice for stress-related queries.
<img width="1512" alt="first_page" src="https://github.com/user-attachments/assets/e1b89ee6-f902-49af-ad90-d395143933cd">

### 2. Mood Logging

- Allows users to log their mood (e.g., Happy, Neutral, Stressed).

- Tracks causes of stress for better self-awareness.
  
### 3. Personalized Stress Reports

- Generates textual and graphical summaries of mood trends.

- Users can download a PDF report summarizing their stress data.
### 4. Feedback System

- Users can provide thumbs-up or thumbs-down feedback for chatbot responses.

- Feedback data is used to analyze and improve the chatbot's responses.

### 5. Visualization

- Visualizes mood trends over time using interactive and static charts.

- Provides insights into stress patterns and common causes.

### 6. Support Between Professional Sessions

- Helps users track their moods, anxiety levels, and causes when they cannot access professionals.

- Provides reports and charts summarizing mood trends during periods without professional support.

- Assists users in remembering and describing how they felt every day over weeks, which can be difficult to recall during therapy sessions.


## Installation

### Prerequisites

- Python 3.8+

- Install required Python packages:
  
    ```python 
    pip install -r requirements.txt
    ```


## Clone the Repository

  ```python
  git clone https://github.com/yourusername/stress-management-chatbot.git
  cd stress-management-chatbot
  ```

## Run the Application
- 1. Start the backend chatbot service:
   ```python
   python main.py
   ```
- 2. Launch the Streamlit app:
  ```python
  streamlit run streamlit.py
  ```

## 
