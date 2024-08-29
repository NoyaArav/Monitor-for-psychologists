import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go

# Placeholder for functions to handle database interactions and data processing
from database_handler import get_patient_info, get_all_patients, add_patient
from embedding_handler import search_similar_sentences
from Project import process_audio_file, generate_sentiment_graph, detect_drastic_changes, identify_topic_of_change


# Sentiment score dictionaries
patient_sentiment_scores = {
    "Despair": -5, "Anger": -4, "Anxiety": -3, "Sadness": -2, "Discomfort": -1,
    "Natural": 0, "Contentment": 1, "Hopefulness": 2, "Happiness": 3, "Excitement": 4, "Euphoria": 5
}

psychologist_sentiment_scores = {
    "Overwhelm": -5, "Helplessness": -4, "Sadness": -3, "Frustration": -2, "Concern": -1,
    "Natural": 0, "Contentment": 1, "Encouragement": 2, "Empathy": 3, "Optimism": 4, "Fulfillment": 5
}

# Main page sidebar for patient selection
st.sidebar.title("Psychological Sessions Monitor")
patients = get_all_patients()
selected_patient = st.sidebar.selectbox("Select a patient", patients)

# Add New Patient functionality
if st.sidebar.button("Add new patient"):
    st.sidebar.title("Add New Patient")

    # Input fields for new patient details
    new_patient_name = st.sidebar.text_input("Enter patient name:")
    new_patient_birthdate = st.sidebar.date_input("Enter birthdate:")
    new_patient_notes = st.sidebar.text_area("Enter notes:")

    if st.sidebar.button("Create Patient"):
        # Determine the next patient_id
        if len(patients) == 0:
            new_patient_id = 1
        else:
            new_patient_id = max(patient['patient_id'] for patient in patients) + 1

        # Add new patient to the database
        add_patient(new_patient_id, new_patient_name, new_patient_birthdate, new_patient_notes)
        st.sidebar.success(f"Patient '{new_patient_name}' added successfully!")


if selected_patient:
    st.sidebar.subheader(f"Patient: {selected_patient}")
    option = st.sidebar.radio("Options", ("Information", "Search", "Sessions", "Add New Session"))
    
    # Information page
    if option == "Information":
        st.title(f"Patient Information")
        
        patient_info = get_patient_info(selected_patient['patient_id'])

        # Display general information about the patient
        st.write(f"**Name:** {patient_info['name']}")
        st.write(f"**Birthdate:** {patient_info['birthdate']}")
        st.write(f"**Notes:** {patient_info['notes']}")
    
        
        # Display sentiment graph for the last session
        st.subheader("Patient's Emotions During Last Session")
        sentiment_graph = generate_sentiment_graph(patient_info['sessions'][-1], "Patient Sentiment Analysis", patient_sentiment_scores)
        st.plotly_chart(sentiment_graph)
        
        # Display topics causing emotional changes
        st.subheader("Topics Which Caused Emotional Changes During Last Session")
        last_session_data = patient_info['sessions'][-1]  # Get data of the last session

        patient_data = [entry for entry in last_session_data if entry['speaker'] == 'patient']
        threshold = 3  # Example threshold for detecting drastic changes

        drastic_changes = detect_drastic_changes(patient_data, threshold)
        topics = []

        for change in drastic_changes:
            id1, id2, change_value = change
            context = [entry for entry in last_session_data if id1 - 3 <= entry['id'] <= id2 + 3]
            sentence_1 = next(item['sentence'] for item in patient_data if item['id'] == id1)
            sentence_2 = next(item['sentence'] for item in patient_data if item['id'] == id2)
            emotion_1 = next(item['sentiment'] for item in patient_data if item['id'] == id1)
            emotion_2 = next(item['sentiment'] for item in patient_data if item['id'] == id2)

            topic = identify_topic_of_change([item['sentence'] for item in context], sentence_1, sentence_2, change_value, emotion_1, emotion_2, "patient")
            topics.append(topic)
        
        if topics:
            st.write("Drastic emotional changes detected, topics causing these changes:")
            for topic in topics:
                st.write(f"- {topic}")
        else:
            st.write("No drastic emotional changes detected in the last session.")

        
        
        # st.write("Topics from previous sessions not discussed in the last session:")
        # st.write(patient_info['topics_previous_sessions'])

    # Search page
    elif option == "Search":
        st.title(f"Search for: {selected_patient['name']}")
        
        patient_info = get_patient_info(selected_patient['patient_id'])
        
        search_query = st.text_input("Type here to search:")
        
        # Generate session options dynamically
        session_options = ['All sessions'] + [f"Session {session['session_id']}" for session in patient_info['sessions']]
        sessions_to_search = st.multiselect("Select sessions to search", options=session_options, default='All sessions')
        
        if st.button("Search"):
            # Convert session options into actual session IDs for searching
            if 'All sessions' in sessions_to_search or not sessions_to_search:
                selected_session_ids = [session['session_id'] for session in patient_info['sessions']]
            else:
                selected_session_ids = [int(option.split()[1]) for option in sessions_to_search]
            
            search_results = search_similar_sentences(selected_patient['patient_id'], search_query, selected_session_ids)
            
            if search_results:
                # Determine if displaying the session number column is needed
                show_session_column = len(selected_session_ids) > 1
                
                # Create table header
                columns = ["Sentence Number", "Sentence", "Speaker"]
                if show_session_column:
                    columns.insert(0, "Session Number")
                    
                # Display results in table format
                st.write("Search Results:")
                for result in search_results:
                    if show_session_column:
                        st.write(f"{result['session_id']} | {result['sentence_number']} | {result['sentence']} | {result['speaker']}")
                    else:
                        st.write(f"{result['sentence_number']} | {result['sentence']} | {result['speaker']}")
            else:
                st.write("No results found.")

    # Sessions page
    elif option == "Sessions":
        st.title(f"Session Transcripts for: {selected_patient['name']}")
        
        patient_info = get_patient_info(selected_patient['patient_id'])
        
        for session in patient_info['sessions']:
            st.subheader(f"Session {session['session_id']}")

            # Display each sentence in the session
            for idx, sentence_data in enumerate(session['session_data'], start=1):
                # Formatting the output
                row_number = sentence_data['id']
                speaker = sentence_data['speaker']
                sentence = sentence_data['sentence']
                
                st.write(f"{idx}. {speaker}: {sentence}")

    # Add New Session page
    elif option == "Add New Session":
        st.title(f"Add New Session for {selected_patient['name']}")
        audio_file = st.file_uploader("Upload session audio file", type=["wav", "mp3", "mp4"])
        
        if st.button("Upload"):
            if audio_file is not None:
                process_audio_file(selected_patient['patient_id'], audio_file)
                st.success("Session added successfully!")
