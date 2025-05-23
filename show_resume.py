import streamlit as st

# Define the pages
main_page = st.Page("find_resume.py", title="Recherche", icon=":material/add_circle:")
page_2 = st.Page("set_prompt.py", title="Modif. prompt", icon=":material/info:")


# Set up navigation
pg = st.navigation([main_page, page_2])

# Run the selected page
pg.run()