from main import Chatbot  # Import your Chatbot class
import streamlit as st

# Define allowed moods
MOODS = [
    "Sad, Depressed", "Guilty, Ashamed", "Angry, Irritated, Annoyed, Resentful",
    "Frustrated", "Anxious, Worried, Terrified, Nervous, Panicked",
    "Inferior, Inadequate", "Lonely", "Hopeless, Discouraged", "Happy", "Neutral"
]

# Initialize the chatbot
try:
    bot = Chatbot()
except Exception as e:
    st.error(f"Error initializing chatbot: {e}")
    st.stop()

# Page configuration
st.set_page_config(page_title="Stress Management Chatbot", layout="wide")

# Main Interface
st.title("Stress Management Chatbot")
st.markdown("Share your concerns or ask questions about stress, and I'll provide empathetic and actionable advice.")

# Initialize session state variables
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []  # Stores the conversation history
if "display_chat" not in st.session_state:
    st.session_state.display_chat = True  # Controls whether chat is displayed

# Input field for user input
user_input = st.text_input("Share your thoughts")

# Handle "Send" button click
if st.button("Send"):
    print("send")
    if user_input.strip():  # Ensure input is not empty
        # If chat was cleared earlier, reset display_chat to True
        if not st.session_state.display_chat:
            st.session_state.display_chat = True

        # Generate bot response
        bot_response = bot.ask_with_history(user_input, st.session_state.conversation_history)

        # Update the conversation history
        st.session_state.conversation_history.append({"user": user_input, "bot": bot_response})

        # Clear the user input field for better UX
        user_input = ""
    else:
        st.warning("Please enter a message to continue the conversation.")

# Add a button to clear the conversation display
if st.button("Clear Conversation"):
    # Hide the chat from the screen
    st.session_state.display_chat = False
    print("clear")
    # The history remains saved in conversation_history
    st.success("Chat cleared from the screen! Conversation is still saved.")

# Conversation Section
st.header("Conversation")
if st.session_state.display_chat:
    if st.session_state.conversation_history:
        for turn in st.session_state.conversation_history:
            st.markdown(f"**You:** {turn['user']}")
            st.markdown(f"**Bot:** {turn['bot']}")
    else:
        st.write("No conversation yet. Start by sharing your thoughts or concerns!")
else:
    st.write("Chat has been cleared from the screen. Start a new conversation or continue.")



# Add feedback buttons for the last bot response
if st.session_state.conversation_history:
    st.write("Was this response helpful?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍", key="thumbs_up"):
            last_turn = st.session_state.conversation_history[-1]
            bot.save_feedback_with_emotion(last_turn['user'], last_turn['bot'], "positive")
            st.success("Thank you for your positive feedback!")
    with col2:
        if st.button("👎", key="thumbs_down"):
            last_turn = st.session_state.conversation_history[-1]
            bot.save_feedback_with_emotion(last_turn['user'], last_turn['bot'], "negative")
            st.warning("Thank you for your feedback!")

# Mood logging section
st.header("Log Your Mood")

# Add a placeholder option to the mood dropdown
mood = st.selectbox(
    "Select your current mood:", 
    options=["Select a mood"] + MOODS  # Placeholder added as the first option
)

# Input for the cause of the mood
cause = st.text_input("What is causing this mood?")

if st.button("Log Mood"):
    if mood != "Select a mood" and cause:  # Ensure a valid mood is selected
        bot.log_mood(mood, cause)
        st.success("Your mood has been logged!")
    elif mood == "Select a mood":
        st.warning("Please select a mood from the dropdown list.")
    else:
        st.warning("Please enter a cause for your mood.")


# PDF Report Generation Section
st.header("Your Personalized Stress Report")
if st.button("Generate PDF Report"):
    pdf_path = bot.generate_pdf_report()
    if pdf_path:
        with open(pdf_path, "rb") as pdf_file:
            pdf_data = pdf_file.read()
            st.download_button(
                label="Download Stress Report as PDF",
                data=pdf_data,
                file_name="stress_report.pdf",
                mime="application/pdf",
            )
    else:
        st.error("Failed to generate the PDF report.")


# Mood Trend Visualization
st.header("Mood Trends")

# Interactive Mood Trend Visualization
if st.button("Show Interactive Mood Trends"):
    fig = bot.plot_interactive_mood_trend()
    if fig:
        st.plotly_chart(fig)
    else:
        st.warning("No mood data available to generate a chart.")
