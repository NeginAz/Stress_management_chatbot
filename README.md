# Stress Management Chatbot

This repository contains a Stress Management Chatbot application designed to provide users with empathetic and actionable advice for managing stress. The chatbot leverages AI, Retrieval-Augmented Generation (RAG), and user feedback to continuously improve its responses and user experience.

## Features

### 1. Chatbot with Multi-Turn Conversations

- Maintains context across multiple user queries.

- Provides concise and empathetic advice for stress-related queries.

![chat](https://github.com/user-attachments/assets/8eeca824-cd35-4207-aac2-31eca3e1ce4f)


### 2. Mood Logging

- Allows users to log their mood (e.g., Happy, Stressed, Depressed).

- Tracks causes of stress for better self-awareness.

![log_moods](https://github.com/user-attachments/assets/a7071e92-e21e-4c7a-a509-316da04edbed)


### 3. Personalized Stress Reports

- Generates textual and graphical summaries of mood trends.

- Users can download a PDF report summarizing their stress data.
### 4. Feedback System

- Users can provide thumbs-up or thumbs-down feedback for chatbot responses.

- Feedback data is used to analyze and improve the chatbot's responses.
![figure](https://github.com/user-attachments/assets/be74237c-26bb-434a-956a-82f130a6a7c8)


### 5. Visualization

- Visualizes mood trends over time using interactive and static charts.

- Provides insights into emotion patterns and common causes.

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


#### Clone the Repository

  ```python
  git clone https://github.com/yourusername/stress-management-chatbot.git
  cd stress-management-chatbot
  ```

#### Run the Application
- 1. Start the backend chatbot service:
   ```python
   python main.py
   ```
- 2. Launch the Streamlit app:
  ```python
  streamlit run streamlit.py
  ```

### 7. Data Source

The chatbot's responses are enhanced using a Retrieval-Augmented Generation (RAG) pipeline, which focuses on information from a custom `stress.txt` file. This file contains carefully curated content from the following sources:

- [Source 1](https://mindbodyo.com/stress-management-techniques/) - Effective Stress Management Techniques for Better Mental Health.
- [Source 2](https://www.mayoclinic.org/healthy-lifestyle/stress-management/in-depth/relaxation-technique/art-20045368) - Holistic Approaches to Stress Management. 
- [Source 3](https://www.mayoclinic.org/healthy-lifestyle/stress-management/in-depth/relaxation-technique/art-20045368) - Stress management. 

The content was selected to ensure the chatbot provides accurate, actionable, and empathetic advice to users.

