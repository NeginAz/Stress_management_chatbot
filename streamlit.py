from main import Chatbot  # Import your Chatbot class
import streamlit as st

from main import Chatbot
import streamlit as st

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
st.markdown("Ask me anything about stress, and I'll provide concise advice to help you manage it.")


# Initialize session state variables
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []  # For storing chat history
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "response" not in st.session_state:
    st.session_state.response = ""



# Input field for user questions
user_input = st.text_input("Your Question:")
if st.button("Send"):
    if user_input:
        # Generate response while considering conversation history
        response = bot.ask_with_history(user_input, st.session_state.conversation_history)

        # Append user input and bot response to the conversation history
        st.session_state.conversation_history.append({"user": user_input, "bot": response})

        # Display the entire conversation
        for turn in st.session_state.conversation_history:
            st.markdown(f"**User**: {turn['user']}")
            st.markdown(f"**Bot**: {turn['bot']}")

        # Update the session state
        st.session_state.user_input = user_input
        st.session_state.response = response
    else:
        st.warning("Please enter a question.")

# Add a button to clear the conversation history
if st.button("Clear Conversation"):
    st.session_state.conversation_history = []  # Reset the chat history
    st.success("Conversation history cleared!")




# # Initialize session state variables
# if "user_input" not in st.session_state:
#     st.session_state.user_input = None
# if "response" not in st.session_state:
#     st.session_state.response = None
# # Input field for user questions
# user_input = st.text_input("Your Question:", key="user_input_field")
# if st.button("Get Response"):
#     if user_input:
#         # Save user input and response in session state
#         st.session_state.user_input = user_input
#         st.session_state.response = bot.ask(user_input)

#         st.success("Response generated!")
#     else:
#         st.warning("Please enter a question.")

# # Display the response
# if st.session_state.response:
#     st.write(f"Response: {st.session_state.response}")



# Add evaluation buttons
if st.session_state.user_input and st.session_state.response:
    st.write("Was this response helpful?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍", key="thumbs_up"):
            bot.save_feedback_with_emotion(
                st.session_state.user_input,
                st.session_state.response,
                "positive"
            )
            st.success("Thank you for your positive feedback!")
    with col2:
        if st.button("👎", key="thumbs_down"):
            bot.save_feedback_with_emotion(
                st.session_state.user_input,
                st.session_state.response,
                "negative"
            )
            st.warning("Thank you for your feedback!")


# Mood logging section
st.header("Log Your Mood")
# Mood selection
mood = st.selectbox("How are you feeling today?", ["Happy", "Neutral", "Stressed"])
cause = st.text_input("What is causing this mood?")

if st.button("Log Mood"):
    if mood and cause:
        bot.log_mood(mood, cause)
        st.success("Your mood has been logged!")
    else:
        st.warning("Please select a mood and enter a cause.")




# Report generation section
st.header("Your Personalized Stress Report")

if st.button("Generate Report"):
    report = bot.generate_report()
    st.markdown(report)



# Mood Trend Visualization
st.header("Mood Trends")

if st.button("Show Mood Trends"):
    chart_path = bot.plot_mood_trend()
    if chart_path:
        st.image(chart_path, caption="Mood Trends Over Time")
    else:
        st.warning("No mood data available to generate a chart.")


# Interactive Mood Trend Visualization
#st.header("Mood Trends")

if st.button("Show Interactive Mood Trends"):
    fig = bot.plot_interactive_mood_trend()
    if fig:
        st.plotly_chart(fig)
    else:
        st.warning("No mood data available to generate a chart.")




# from main import Chatbot  # Import your Chatbot class
# import streamlit as st

# # Initialize the chatbot
# bot = Chatbot()

# # Set up the Streamlit page configuration
# st.set_page_config(page_title="Stress Management Chatbot", layout="wide")
# with st.sidebar:
#     st.title("Stress Management Chatbot")
#     st.markdown("Ask me anything about stress, and I'll provide insights to help you manage it.")

# # Function to generate responses
# def generate_response(user_input):
#     #return bot.ask(user_input)
#     raw_response = bot.ask(user_input)
#     #clean up the response if necessary
#     clean_response = raw_response.split("Answer:")[-1].strip()  # Remove unwanted prompt parts
#     return clean_response

# # Initialize session state for storing messages
# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {"role": "assistant", "content": "Welcome! Ask me anything about stress."}
#     ]

# # Display the chat messages
# for message in st.session_state.messages:
#     if message["role"] != "system":  # Hide system messages
#         with st.chat_message(message["role"]):
#             st.write(message["content"])

# # Accept user input
# if user_input := st.chat_input("Type your question here..."):
#     # Add user's message to the chat
#     st.session_state.messages.append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.write(user_input)

#     # Generate assistant's response
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             response = generate_response(user_input)
#             st.write(response)
#     # Add assistant's response to the chat
#     st.session_state.messages.append({"role": "assistant", "content": response})
