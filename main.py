"""
LinkedIn Post Generator — powered by LangChain + Groq
Run with: streamlit run app.py
"""

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
    # Professional Updates
    "New job announcement", "Work anniversary", "Achievement / milestone", 
    "Promotion", "Hiring announcement", "Company layoff response",
    # Thought Leadership & Insights
    "Thought leadership", "Lessons learned / failure story", "Industry trends analysis",
    "Tips / advice", "Book / article review", "Myth busting",
    # Education & Personal Growth
    "Certification completed", "Graduation", "Course recommendation", "Personal transformation",
    # Events & Networking
    "Conference / event recap", "Team appreciation", "Networking / gratitude", 
    "Event invitation", "Webinar announcement",
    # Business & Product
    "Product launch", "Startup journey update", "Funding round announcement", 
    "Case study / client success", "Behind-the-scenes look"
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

# ---------------------------------------------------------------------------
# Sidebar Settings & Website Directory List
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Groq API Key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Get a free key at https://console.groq.com/keys. "
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
    
    # Curated External Web Resources Directory
    st.title("🌐 Website Directory")
    st.subheader("Social Platforms")
    st.markdown("- [LinkedIn](https://linkedin.com)")
    st.markdown("- [Twitter / X](https://x.com)")
    st.markdown("- [Medium](https://medium.com)")
    st.markdown("- [Threads](https://threads.net)")
    
    st.subheader("Tech & Startup Hubs")
    st.markdown("- [Product Hunt](https://producthunt.com)")
    st.markdown("- [Hacker News](https://ycombinator.com)")
    st.markdown("- [GitHub](https://github.com)")
    
    st.subheader("AI Resources")
    st.markdown("- [OpenAI](https://openai.com)")
    st.markdown("- [Anthropic](https://anthropic.com)")
    st.markdown("- [Streamlit Docs](https://streamlit.io)")

# ---------------------------------------------------------------------------
# Main Layout UI
# ---------------------------------------------------------------------------
# Header Row
img_col, title_col = st.columns([1, 4])
with img_col:
    # Safe check/fallback if image asset isn't present during rapid local deploys
    try:
        st.image("ChatGPT.png", width=120)
    except Exception:
        st.text("📝 LIF.ai Logo")

with title_col:
    st.title("LIF.ai")
    st.caption("Powered by LangChain + Groq — turn a rough idea into a polished LinkedIn post, Instagram and facebook in seconds.")

st.markdown("---")

# Input Configuration Selectors
col1, col2 = st.columns(2)
with col1:
    category = st.selectbox("Post type", CATEGORIES)
with col2:
    tone = st.selectbox("Tone", TONES)

topic = st.text_area(
    "Topic — what's the post about?",
    placeholder="e.g. I just shipped a new feature that reduced load times by 40%...",
    height=120,
)

# Output Style Controls
col3, col4, col5 = st.columns(3)
with col3:
    length = st.selectbox("Length", ["Short", "Medium", "Long"], index=1)
with col4:
    language = st.selectbox("Language", LANGUAGES, index=0)
with col5:
    st.write("**Formatting Toggles**")
    use_emojis = st.checkbox("Emojis", value=True)
    use_hashtags = st.checkbox("Include hashtags", value=True)

# Optional few-shot dataset switch
use_examples = st.checkbox(
    "Use style examples from linkedin_post_dataset.json (if present in this folder)",
    value=False,
)

# ---------------------------------------------------------------------------
# Few-shot example loader
# ---------------------------------------------------------------------------
def load_examples(category_name: str, n: int = 2):
    path = os.path.join(os.path.dirname(__file__), "linkedin_post_dataset.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    
    slug = category_name.lower().split(" ")[0]
    matches = [d for d in data if slug in d.get("category", "")]
    if not matches:
        matches = data
    return matches[:n]

LENGTH_MAP = {
    "Short": "under 80 words",
    "Medium": "120-180 words",
    "Long": "220-300 words",
}

# ---------------------------------------------------------------------------
# LangChain prompt + chain
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert LinkedIn ghostwriter. You write authentic, engaging, "
    "human-sounding LinkedIn posts that avoid generic corporate language and "
    "clickbait. Return only the finished post text, with no preamble, "
    "explanation, or surrounding quotation marks."
)

USER_PROMPT_TEMPLATE = """Write a LinkedIn, Instagram, Facebook and E book post of type '{category}'. 
Tone: {tone}. 
Length: {length_desc}. 
Language: write the ENTIRE post in {language}. 

Use short paragraphs and line breaks for readability, the way real LinkedIn posts are formatted. 
{emoji_instruction} 
{hashtag_instruction} 

Topic / idea to base the post on: {topic} 

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
                        "category": category,
                        "tone": tone,
                        "length_desc": LENGTH_MAP[length],
                        "language": language,
                        "emoji_instruction": emoji_instruction,
                        "hashtag_instruction": hashtag_instruction,
                        "topic": topic.strip(),
                        "example_block": example_block,
                    }
                ).strip()
                st.session_state.history.insert(0, post_text)
        except Exception as e:
            st.error(f"Something went wrong calling Groq via LangChain: {e}")

# ---------------------------------------------------------------------------
# App Generation Rendering View
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.markdown("### Generated post")
    latest = st.session_state.history[0]
    st.text_area("Post preview", latest, height=280, label_visibility="collapsed")
    st.download_button(
        "⬇ Download as .txt",
        data=latest,
        file_name="linkedin_post.txt",
        mime="text/plain",
    )
    
    if len(st.session_state.history) > 1:
        with st.expander("Previous generations"):
