#Author: Suhail Jamaldeen

# Core imports
import streamlit as st

#Author: Suhail Jamaldeen

# Initialize session state for history if not already set
if 'question_history' not in st.session_state:
    st.session_state.question_history = []


# Load external CSS (if present) so styles live in `assets/styles.css`
css_path = "assets/styles.css"
try:
    with open(css_path, "r") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)
except Exception:
    # If CSS can't be read, proceed without it
    pass

# Display logo and smaller header centered using a 3-column layout
console_log("text_to_log")
cols = st.columns([1, 2, 1])
with cols[1]:
    try:
        st.image("assets/arabiers.png", width=200)
    except Exception:
        # If the image can't be shown, continue silently
        pass

    # Use the external CSS class for the header
    st.markdown("<div class='prizo-header'>Prizo AI Agent</div>", unsafe_allow_html=True)
# Input field for user question
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