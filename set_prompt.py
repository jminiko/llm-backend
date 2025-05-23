import streamlit as st
from streamlit.components.v1 import html
from st_paywall import add_auth

with st.sidebar:
    st.write("abonnez vous pour moins de 30€ / mois et recherchez dans notre base de plus 35 000 CVs")
    html('<script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="21talents" data-color="#FFDD00" data-emoji=""  data-font="Cookie" data-text="Buy me a coffee" data-outline-color="#000000" data-font-color="#000000" data-coffee-color="#ffffff" ></script>')
    add_auth(required=True)
    st.write(f"vous êtes: {st.session_state.email}")
prompt = ""
with open(".streamlit/prompt.cfg", "r") as f:
    prompt = f.read()


st.header("Modif. prompt")

prompt_resume = st.text_area("Prompt",value=prompt,height=300)

if st.button("Sauver"):
    with open(".streamlit/prompt.cfg", "w") as f:
        f.write(prompt_resume)

if __name__ == '__main__':
    main()