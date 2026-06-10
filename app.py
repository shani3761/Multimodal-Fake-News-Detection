import streamlit as st

st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("🔍 Multimodal Fake News Detection System")
st.subheader("BSCS 7th Semester FYP - Abbottabad University")

st.success("✅ Application Loaded Successfully!")

st.write("### Choose Input Type")
option = st.radio("Select One:", ["Text Analysis", "Image Analysis", "Video URL Analysis"])

if st.button("🚀 Analyze Now"):
    st.balloons()
    st.info("This is a working demo version. Full AI models will be connected soon.")

st.caption("Developed by Your Name | AUST")