import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from streamlit_chat import message
import torch
import time

# إعداد صفحة Streamlit
st.set_page_config(page_title="CartBot-AI 🤖🛒", page_icon="🛒", layout="centered")

# إعداد Hugging Face Token والموديل
hf_token = "YOUR_HUGGINGFACE_TOKEN"  # <-- ضع هنا التوكن بتاعك
model_name = "microsoft/phi-4"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 torch_dtype=torch.float16, 
                                                 device_map="auto", 
                                                 use_auth_token=hf_token)
    return tokenizer, model

tokenizer, model = load_model()

# تخزين الجلسة
if "history" not in st.session_state:
    st.session_state.history = []

# عنوان التطبيق
st.title("🤖🛒 CartBot-AI")

st.write("""
مرحبًا بك في CartBot-AI!  
مساعدك الذكي للتحدث، التفكير، وإنجاز المهام بكل سهولة.  
ابدأ المحادثة بالأسفل! 👇
""")

# حقل إدخال المستخدم
user_input = st.chat_input("اكتب رسالتك هنا...")

# معالجة إدخال المستخدم
if user_input:
    with st.spinner("CartBot-AI يفكر... 🤔"):
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
        bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # تخزين الرسائل في الجلسة
    st.session_state.history.append((user_input, bot_response))

# عرض المحادثة بشكل مرتب
for i, (user_msg, bot_msg) in enumerate(st.session_state.history):
    message(user_msg, is_user=True, key=f"user_{i}", avatar_style="adventurer")
    
    # عرض رد البوت مع كتابة تدريجية (أنيميشن بسيط)
    with st.chat_message("assistant"):
        full_response = ""
        for word in bot_msg.split():
            full_response += word + " "
            st.markdown(full_response)
            time.sleep(0.02)  # سرعة الكتابة (ممكن تخليها أبطأ لو عايز احساس typing أقوى)

