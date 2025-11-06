import streamlit as st
from agent import generate_quotation

# Initialize session state for history if not already set
if 'question_history' not in st.session_state:
    st.session_state.question_history = []

st.header("🛎️ Prizo AI Agent")
st.write("Ask about hotel rooms, availability, pricing, amenities, or meals.")

# Input field for user question
question = st.text_input("Your hotel question (e.g., 'Tell me what we can see in Yala National Park?')")

# Button to submit question
if st.button("Ask Prizo AI"):
    if question:
        with st.spinner("Prizo AI is checking documents..."):
            response = generate_quotation(question)
            # Store question and response in history
            st.session_state.question_history.append({"question": question, "response": response})
            # Display the latest response
            st.markdown("### Prizo AI Response")
            st.markdown(response)
    else:
        st.warning("Please enter a question.")

# Button to clear history
if st.button("Clear History"):
    st.session_state.question_history = []
    st.success("History cleared!")

# Display question history
if st.session_state.question_history:
    st.markdown("### Question History")
    for i, entry in enumerate(st.session_state.question_history):
        with st.expander(f"Question {i+1}: {entry['question'][:50]}..."):
            st.markdown(f"**Question**: {entry['question']}")
            st.markdown(f"**Response**: {entry['response']}")