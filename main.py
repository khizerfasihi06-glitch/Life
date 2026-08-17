import os
import json
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LIF.ai",
    page_icon="🚀",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Expanded Option Lists
# ---------------------------------------------------------------------------
CATEGORIES = [
    "New job announcement", "Work anniversary", "Achievement / milestone", 
    "Promotion", "Hiring announcement", "Company layoff response",
    "Thought leadership", "Lessons learned / failure story", "Industry trends analysis",
    "Tips / advice", "Book / article review", "Myth busting",
    "Certification completed", "Graduation", "Course recommendation", "Personal transformation",
    "Conference / event recap", "Team appreciation", "Networking / gratitude", 
    "Event invitation", "Webinar announcement",
    "Product launch", "Startup journey update", "Funding round announcement", 
    "Case study / client success", "Behind-the-scenes look", "about news update post on media","New update in website"
]

TONES = [
    "Professional", "Casual & friendly", "Inspirational", "Storytelling", "Bold & confident",
    "Humorous & witty", "Empathetic & vulnerable", "Educational & authoritative", 
    "Contrarian / provocative", "Urgent & hype-building"
]

LANGUAGES = [
    "English", "Urdu", "Roman Urdu (Urdu written in English letters)", "Arabic", 
    "Spanish", "French", "Hindi", "German", "Portuguese", "Turkish", 
    "Indonesia", "Russian", "Persian",
]

WEBSITE = [
    "Linkedin", "Facebook", "Instagram", "Youtube", "X (Twitter)", 
    "Threads", "Tiktok", "Snapchat", "Pinterest", "Reddit", 
    "Quora", "Discord", "Whatsapp", "Telegram"
]

# ---------------------------------------------------------------------------
# Sidebar Settings & Website Directory List
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Groq API Key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Get a free key at https://groq.com. "
             "You can also set it as the GROQ_API_KEY environment variable "
             "instead of pasting it here.",
    )
    
    model = st.selectbox(
        "Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
        ],
        index=0,
    )
    
    temperature = st.slider("Creativity (temperature)", 0.0, 1.5, 0.8, 0.1)
    
    st.markdown("---")
    
    st.title(" Website Directory")
    st.subheader("Social Platforms")
    st.markdown("- [LinkedIn](https://linkedin.com)")
    st.markdown("- [Twitter / X](https://x.com)")
    st.markdown("- [Medium](https://medium.com)")
    st.markdown("- [Threads](https://threads.net)")
    
    st.subheader("Tech & Startup Hubs")
    st.markdown("- [GitHub](https://github.com)")
    
    st.subheader("AI Resources")
    st.markdown("- [OpenAI](https://openai.com)")
    st.markdown("- [Anthropic](https://anthropic.com)")

img_col, title_col = st.columns([1, 4])
with img_col:
    try:
        st.image("ChatGPT.png", width=120)
    except Exception:
        st.text("📝 LIF.ai Logo")

with title_col:
    st.title("LIF.ai")
    st.caption("Powered by LangChain + Groq — turn a rough idea into a polished marketing post in seconds.")

st.markdown("---")

# Input Configuration Selectors (Fixed Column matching 3 vs 3)
col1, col2, col3 = st.columns(3)
with col1:
    category = st.selectbox("Post type", CATEGORIES)
with col2:
    tone = st.selectbox("Tone", TONES)
with col3:
    website = st.selectbox("Web platform target", WEBSITE, index=0) # Fixed typo 'seletbox'

topic = st.text_area(
    "Topic — what's the post about?",
    placeholder="e.g. I just shipped a new feature that reduced load times by 40%...",
    height=120,
)

# Output Style Controls (Fixed duplicate col3 variable naming conflicts)
col_len, col_lang, col_opts = st.columns(3)
with col_len:
    length = st.selectbox("Length", ["Short", "Medium", "Long"], index=1)
with col_lang:
    language = st.selectbox("Language", LANGUAGES, index=0)
with col_opts:
    st.write("**Formatting Toggles**")
    use_emojis = st.checkbox("Emojis", value=True)
    use_hashtags = st.checkbox("Include hashtags", value=True)

# Optional few-shot dataset switch
use_examples = st.checkbox(
    "Use style examples from linkedin_post_dataset.json (if present in this folder)",
    value=False,
)

def load_examples(category_name: str, n: int = 2):
    path = os.path.join(os.path.dirname(__file__), "linkedin_post_dataset.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    
    # Fixed matching bug: checking flat string sub-matches safely
    category_slug = category_name.lower()
    matches = [d for d in data if category_slug in d.get("category", "").lower()]
    if not matches:
        matches = data
    return matches[:n]

LENGTH_MAP = {
    "Short": "under 80 words",
    "Medium": "120-180 words",
    "Long": "220-360 words",
}

# Fixed: System and User prompt configurations are streamlined to reference selected platform variable 
SYSTEM_PROMPT = (
    "You are an expert copywriter and social media ghostwriter. You write authentic, engaging, "
    "human-sounding content tailored perfectly for the requested platform. Avoid generic corporate "
    "clichés or extreme clickbait patterns. Return ONLY the finished post text, with no preamble, "
    "no conversational transition chat, and no surrounding quotation marks."
)

USER_PROMPT_TEMPLATE = """Write an optimization-driven post specifically designed for {website}.
Post Category Type: {category}
Desired Tone: {tone}. 
Target Length: {length_desc}. 
Language Constraint: Write the ENTIRE text exclusively in {language}. 

Formatting Rules:
- Use clean layout patterns, short paragraphs, and distinct line breaks to maximize readability.
- {emoji_instruction} 
- {hashtag_instruction} 

Core Content Context Topic:
{topic} 

{example_block}"""

def build_chain(api_key: str, model: str, temperature: float):
    llm = ChatGroq(api_key=api_key, model=model, temperature=temperature)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT_TEMPLATE),
        ]
    )
    return prompt | llm | StrOutputParser()

# ---------------------------------------------------------------------------
# Generation Logic Engine
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

generate = st.button("🚀 Generate post", type="primary", use_container_width=True)

if generate:
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar (or set GROQ_API_KEY).")
    elif not topic.strip():
        st.error("Please describe what the post should be about.")
    else:
        emoji_instruction = (
            "Include a few relevant emojis, used sparingly." if use_emojis else "Do not use any emojis."
        )
        hashtag_instruction = (
            "End with 3-5 relevant hashtags." if use_hashtags else "Do not include hashtags."
        )
        
        example_block = ""
        if use_examples:
            examples = load_examples(category)
            if examples:
                joined = "\n\n".join(f"Example post:\n{ex['post']}" for ex in examples)
                example_block = (
                    "Here are a couple of example posts in a similar style for reference "
                    "(do not copy them, just match the tone and structure):\n\n" + joined
                )
        
        try:
            with st.spinner("Writing your post..."):
                chain = build_chain(api_key, model, temperature)
                post_text = chain.invoke(
                    {
                        "website": website,
                        "category": category,
                        "tone": tone,
                        "length_desc": LENGTH_MAP[length],
                        "language": language,
                        "emoji_instruction": emoji_instruction,
                        "hashtag_instruction": hashtag_instruction,
                        "topic": topic,
                        "example_block": example_block
                    }
                )
                
                # Display output window
                st.subheader(" Generated Output")
                st.text_area("Copy your post text:", value=post_text, height=350)
                
                # Append to active session history state
                st.session_state.history.append({"platform": website, "content": post_text})
                
        except Exception as e:
            st.error(f"An error occurred during LLM text generation: {e}")

# Render basic log list of generated posts if any exist
if st.session_state.history:
    st.markdown("---")
    st.subheader("📚 Generation History Session Logs")
    for idx, historical_post in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Post {len(st.session_state.history) - idx} - Platform: {historical_post['platform']}"):
            st.code(historical_post['content'], language="text")
