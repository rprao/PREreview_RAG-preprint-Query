import streamlit as st
from app_assets.PREreview_query_engine import PREreview_engine
import re
from app_assets.build_user_retriverquery_engine import build_user_retriverquery_engine
from streamlit_drawable_canvas import st_canvas
import nest_asyncio
nest_asyncio.apply()
import nltk
nltk.download('punkt')


# Apply custom CSS 
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: rgba(255, 255, 255, 0.1); /* Semi-transparent background */
        border-radius: 10px; /* Rounded corners */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Soft shadow */
        backdrop-filter: blur(10px); /* Apple-like blur effect */
    }
    .st-emotion-cache-7mitxc {
        position: relative;
        top: 2px;
        background-color: rgba(222, 227, 237, 0.95); /* Light background color */
        z-index: 999991;
        min-width: 244px;
        max-width: 477.9px;
        transform: none;
        transition: transform 300ms, min-width 300ms, max-width 300ms;
        border-radius: 15px; /* Rounded corners for a softer look */
        padding: 10px; /* Add padding for content */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2); /* Subtle shadow for a card-like effect */
        font-family: 'Comic Sans MS', cursive, sans-serif; /* Fun font style */
        color: #2C3E50; /* Darker text for contrast */
    }
    /* Scrollable container for query response */
    .scrollable-container {
        max-height: 400px; /* Set a fixed height */
        overflow-y: scroll; /* Enable vertical scrolling */
        padding: 15px;
        background-color: rgba(255, 255, 255, 0.22); /* Slightly transparent background */
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    /* Hover effect: Background color changes and scale increases slightly for tactile feedback */
    .scrollable-container:hover {
        background-color: rgba(255, 255, 255, 1); /* Make the background fully opaque on hover */
        transform: scale(1.02); /* Slightly scale up for a tactile feel */
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2); /* Stronger shadow on hover */
    }
    </style>
""", unsafe_allow_html=True)

if 'PREreview_query_engine' not in st.session_state:
    st.session_state['PREreview_query_engine'] = PREreview_engine()
if 'user_interacted' not in st.session_state:
    st.session_state['user_interacted'] = False
# Sidebar
with st.sidebar.container():
    st.markdown(" ## About:\n"  
                "This platform enables constructive feedback on preprints by providing a workflow for reviewers to find, review, and publish their reviews.\n"
                " It aims to serve preprint authors, reviewers, and servers globally, promoting equity, diversity, and inclusion in scientific research.")
    PREreview_user_input = st.chat_input("Enter any query you have about PREreview :")
# If the user types something, set the interaction flag to True
if PREreview_user_input:
    st.session_state['user_interacted'] = True

# Sidebar placeholder for query processing
sidebar_placeholder = st.sidebar.empty()

# Handle the query logic
if PREreview_user_input:
    with st.sidebar.container():
        sidebar_placeholder.write("Processing your query and self-correcting my response...")
        output_response = st.session_state['PREreview_query_engine'].query(PREreview_user_input)
        sidebar_placeholder.empty()  # Clear the placeholder after query is processed
        st.markdown(f"<div class='scrollable-container'>{str(output_response)}</div>",unsafe_allow_html=True)
elif st.session_state['user_interacted']:
    st.sidebar.info('e.g How to connect with PREreview')
    
with st.sidebar.container():
    st.markdown("## A place for your notes")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width= 2,
        stroke_color="black",
        background_color="#F3CECE",
        background_image= None,
        update_streamlit=True,
        height=300,
        width=500,
        drawing_mode="freedraw",
        point_display_radius= 0,
        key="canvas",
        )
#---------------------------------------------------------------------------
#Above give logic is for side bar were they can acess any information on the PREreview
logo_path = "logo.png"
st.image(logo_path, width=400)

user_input = st.chat_input("Enter your preprint PDF link to query on them:")

# Helper function to detect if input is a PDF link based on keywords
def is_pdf_link(text):
    keywords = ['.pdf','PDF','pdf']
    if re.match(r'^https?:\/\/', text):  # Basic URL validation
        for keyword in keywords:
            if keyword in text.lower():
                return True
    return False
#session states 

if 'query_engine' not in st.session_state:
    st.session_state['query_engine'] = None
    
def process_pdf_link(pdf_link):
    placeholder = st.empty()
    placeholder.write(f"Processing PDF link: {pdf_link}\n"
                       "building the index and engine") 
    query_engine = build_user_retriverquery_engine(pdf_link=pdf_link)
    placeholder.empty() 
    return query_engine

#If the user provides a PDF link
if user_input and is_pdf_link(user_input):  # Checking if it's a PDF link
    st.session_state['query_engine'] = process_pdf_link(user_input)
    st.write("Ready! Lets watch your preprint spill all its secrets! Go ahead, give it a try")
    
# Step 5: If the user provides a query after PDF processing
elif user_input and st.session_state['query_engine']:
    placeholder = st.empty()
    placeholder.write("Processing query...")
    query_engine = st.session_state['query_engine']
    pdf_output_response = query_engine.query(user_input)  # Query the engine with the user's query
    placeholder.empty()
    st.markdown(f"<div class='scrollable-container'>{str(pdf_output_response)}</div>",unsafe_allow_html=True)
    st.markdown("To submit a peer-review for this document\n"
             "visit https://prereview.org/review-a-preprint")
# Step 6: If no PDF or query, prompt the user
else:
    st.write("Please provide a PDF link, to begin asking questions")
