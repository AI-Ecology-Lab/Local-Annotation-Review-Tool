import streamlit as st
import os
from pages import run_inference, annotation_review

st.set_page_config(
    page_title="Local Annotation Review",
    page_icon="🔍",
    layout="wide"
)

def main():
    st.title("Local Annotation Review Tool")
    
    # Sidebar navigation
    page = st.sidebar.radio("Navigation", ["Run Inference", "Annotation Review"])
    
    if page == "Run Inference":
        run_inference.show()
    else:
        annotation_review.show()

if __name__ == "__main__":
    main()
