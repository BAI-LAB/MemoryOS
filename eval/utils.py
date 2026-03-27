import time
import uuid
import numpy as np
from openai import OpenAI

def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def generate_id(prefix="id"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def normalize_vector(vec):
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

class OpenAIClient:
    def __init__(self, api_key, base_url=None,
                 embedding_api_key=None, embedding_base_url=None, embedding_model="text-embedding-3-small"):
        self.api_key = api_key
        self.base_url = base_url if base_url else "https://api.openai.com/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.embedding_model = embedding_model
        _emb_key = embedding_api_key or self.api_key
        _emb_url = embedding_base_url or self.base_url
        self.embedding_client = OpenAI(api_key=_emb_key, base_url=_emb_url)
        self._embedding_cache = {}

    def chat_completion(self, model, messages, temperature=0.7, max_tokens=2000):
        print("调用 GPT 接口，模型:", model)
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

    def get_embedding(self, text):
        cache_key = f"{self.embedding_model}::{hash(text)}"
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        response = self.embedding_client.embeddings.create(model=self.embedding_model, input=text)
        embedding = np.array(response.data[0].embedding, dtype=np.float32)

        self._embedding_cache[cache_key] = embedding
        if len(self._embedding_cache) > 10000:
            keys_to_remove = list(self._embedding_cache.keys())[:1000]
            for key in keys_to_remove:
                self._embedding_cache.pop(key, None)
        return embedding

def gpt_generate_answer(prompt, messages, client):
    return client.chat_completion(model="gpt-4o-mini", messages=messages, temperature=0.7, max_tokens=2000)

def analyze_assistant_knowledge(dialogs, client):
    """
    Analyzes conversations to extract knowledge or identity traits about the assistant.
    Returns: {"assistant_knowledge": str}
    """
    conversation = "\n".join([f"User: {d['user_input']}\nAI: {d['agent_response']}\nTime:{d['timestamp']}\n" for d in dialogs])

    prompt = """
# Assistant Knowledge Extraction Task
Analyze the conversation and extract any fact or identity traits about the assistant. 
If no traits can be extracted, reply with "None". Use the following format for output:
The generated content should be as concise as possible — the more concise, the better.
【Assistant Knowledge】
- [Fact 1]
- [Fact 2]
- (Or "None" if none found)

Few-shot examples:
1. User: Can you recommend some movies.
   AI: Yes, I recommend Interstellar.
   Time: 2023-10-01
   【Assistant Knowledge】
   - I recommend Interstellar on 2023-10-01.

2. User: Can you help me with cooking recipes?
   AI: Yes, I have extensive knowledge of cooking recipes and techniques.
   Time: 2023-10-02
   【Assistant Knowledge】
   - I have cooking recipes and techniques on 2023-10-02.

3. User: That’s interesting. I didn’t know you could do that.
   AI: I’m glad you find it interesting!
   【Assistant Knowledge】
   - None

Conversation:
""" + conversation

    messages = [
        {
            "role": "system",
            "content": """You are an assistant knowledge extraction engine. Rules:
1. Extract ONLY explicit statements about the assistant's identity or knowledge.
2. Use concise and factual statements in the first person.
3. If no relevant information is found, output "None".""" 
        },
        {"role": "user", "content": prompt}
    ]

    print("Analyzing assistant knowledge...")
    result = gpt_generate_answer(prompt, messages, client)
    
    # Parse output
    assistant_knowledge = result.replace("【Assistant Knowledge】", "").strip()
    return {"assistant_knowledge": assistant_knowledge}

def gpt_summarize(dialogs, client):
    prompt = "Please generate a topic summary based on the following conversation：\n"
    for d in dialogs:
        prompt += f"user: {d.get('user_input','')}\nassiant: {d.get('agent_response','')}\n"
    prompt += "\nSubject Summary："
    messages = [
        {"role": "system", "content": "You are an expert in summarizing dialogue topics, please generate a concise and precise summary."},
        {"role": "user", "content": prompt}
    ]
    print("调用 GPT 生成主题摘要...")
    return gpt_generate_answer(prompt, messages, client)

def gpt_generate_multi_summary(text, client):
    """
    调用 LLM 生成多子主题摘要，返回格式示例如下：
    {
      "input": "对话文本",
      "summaries": [
         {"theme": "出差", "keywords": ["出差", "行程", "工作"], "content": "用户提到出差相关的困扰"},
         {"theme": "健康", "keywords": ["感冒", "难受", "生病"], "content": "用户反馈感冒导致身体不适"}
      ]
    }
    """
    prompt = ("Please analyze the following dialogue and generate multiple subtopic summaries (if applicable), with a maximum of two themes.\n"
              "Each summary should include the subtopic name, keywords (separated by commas), and the summary text, formatted as a JSON array, with an example format as follows:\n"
              "[\n  {\"theme\": \"Business trip\", \"keywords\": [\"Business trip\", \"Itinerary\", \"Work\"], \"content\": \" User mentioned the troubles related to business trips.\"},\n  {\"theme\": \"Health\", \"keywords\": [\"Cold\", \"Uncomfortable\", \"Sick\"], \"content\": \"User reported feeling unwell due to a cold.\"}\n]\n"
              "Please directly output the JSON array, without adding any other content.\n\Conversation content:\n" + text)
    messages = [
        {"role": "system", "content": "You are an expert in analyzing dialogue topics. No more than two topics."},
        {"role": "user", "content": prompt}
    ]
    print("调用 GPT 生成多子主题摘要...")
    response_text = gpt_generate_answer(prompt, messages, client)
    import json
    try:
        summaries = json.loads(response_text)
    except Exception:
        summaries = []
    return {"input": text, "summaries": summaries}

# def gpt_personality_analysis(dialogs, client):
#     prompt = ("Please analyze the following conversation and extract the user profile information and user private data."
#               "Please output in the following format:\n"
#               "【User Profile】\n"
#               "Areas of Interest:\n"
#               "Response Preferences：\n"
#               "Preferred Content Type：\n"
#               "Short vs. Detailed Responses：\n"
#               "Formal vs. Casual Tone：\n"
#               "Other Notes:：\n"
#               "【User Private Data】\n"
#               "Please list all the private information involved (such as account numbers, passwords, user purchase,etc.). If there is none, please write \"None\"\n\n"
#               "The conversation is as follows:\n")
#     for d in dialogs:
#         prompt += f"User: {d.get('user_input','')}\nAssiant: {d.get('agent_response','')}\n"
#     messages = [
#         {"role": "system", "content": "You are a professional user profile analyst who can also identify user private data. Please strictly follow the template for output."},
#         {"role": "user", "content": prompt}
#     ]
#     print("调用 GPT 分析用户画像和私有数据...")
#     result_text = gpt_generate_answer(prompt, messages, client)
#     profile, private = "", ""
#     parts = result_text.split("【User Private Data】")
#     if len(parts) == 2:
#         profile = parts[0].replace("【User Profile】", "").strip()
#         private = parts[1].strip()
#     else:
#         profile = result_text.strip()
#         private = "None"
#     return {"profile": profile, "private": private}
# def gpt_personality_analysis(dialogs, client):
#     """
#     Analyzes conversations to extract structured personality traits, private knowledge, 
#     and assistant-related knowledge.
#     Returns: {"profile": str, "private": str, "assistant_knowledge": str}
#     """
#     conversation = "\n".join([f"User: {d['user_input']}\nAssistant: {d['agent_response']}" for d in dialogs])

#     prompt = """
# # Personality Analysis Task
# Analyze the conversation and output in EXACTLY this format:

# 【User Profile】
# 1. Core Psychological Traits:
#    - [Trait]: [Positive/Negative/Neutral] (Evidence)
#    - (Max 5 most prominent traits)

# 2. Content Preferences:
#    - [Topic]: [Like/Dislike/Neutral] (Evidence)
#    - (Max 5 strongest preferences)

# 3. Interaction Style:
#    - [Style]: [Preference] (Evidence)
#    - (e.g., Direct/Indirect, Detailed/Concise)

# 4. Value Alignment:
#    - [Value]: [Strong/Weak] (Evidence)
#    - (e.g., Honesty, Helpfulness)

# 【User Private Data】
# - [Fact 1]
# - [Fact 2]
# - (Or "None" if none found)

# Conversation:
# """ + conversation

#     messages = [
#         {
#             "role": "system",
#             "content": """You are a personality analysis engine. Rules:
# 1. Extract ONLY observable traits with direct evidence
# 2. Use standardized trait names from psychology
# 3. Mark confidence: Positive=explicit preference, Neutral=implied
# 4. Private data includes possessions, habits, and sensitive preferences"""
#         },
#         {"role": "user", "content": prompt}
#     ]

#     print("Running personality analysis...")
#     result = gpt_generate_answer(prompt, messages, client)
    
#     # Parse output
#     profile, private = result.split("【User Private Data】") if "【User Private Data】" in result else (result, "None")
    
#     # Analyze assistant knowledge
#     assistant_knowledge_result = analyze_assistant_knowledge(dialogs, client)
    
#     return {
#         "profile": profile.replace("【User Profile】", "").strip(),
#         "private": private.strip(),
#         "assistant_knowledge": assistant_knowledge_result["assistant_knowledge"]
#     }
def gpt_personality_analysis(dialogs, client):
    """
    Analyzes conversations to extract structured personality traits, general user data, 
    and assistant-related knowledge.
    Returns: {"profile": str, "user_data": str, "assistant_knowledge": str}
    """
    conversation = "\n".join([f"User: {d['user_input']}\nAssistant: {d['agent_response']}\nTime:{d['timestamp']}" for d in dialogs])

    prompt = """
# Personality and User Data Analysis Task
Analyze the conversation and output in EXACTLY this format:

【User Profile】
1. Core Psychological Traits:
   - [Trait]: [Positive/Negative/Neutral] (Evidence)
   - (Max 5 most prominent traits)

2. Content Preferences:
   - [Topic]: [Like/Dislike/Neutral] (Evidence)
   - (Max 5 strongest preferences)

3. Interaction Style:
   - [Style]: [Preference] (Evidence)
   - (e.g., Direct/Indirect, Detailed/Concise)

4. Value Alignment:
   - [Value]: [Strong/Weak] (Evidence)
   - (e.g., Honesty, Helpfulness)

【User Data】
- [Fact 1]: [Details] (e.g., "User mentioned visiting a park on April 1st, 2025 in New York.")
- [Fact 2]: [Details] (e.g., "User likes pizza, enjoys sci-fi movies, and dislikes rainy weather.")
- (Include events, dates, locations, preferences, or other general or private information explicitly mentioned in the conversation. If none, write "None.")

Conversation:
""" + conversation
    messages = [
        {
            "role": "system",
            "content": """You are a personality and user data analysis engine. Rules:
1. Extract ONLY observable traits and data with direct evidence.
2. Include general user data such as events, dates, locations, and preferences.
3. Use concise and factual statements.
4. If no relevant information is found, output "None"."""
        },
        {"role": "user", "content": prompt}
    ]

    print("Running personality and user data analysis...")
    result = gpt_generate_answer(prompt, messages, client)
    
    # Parse output
    profile, user_data = result.split("【User Data】") if "【User Data】" in result else (result, "None")
    
    # Analyze assistant knowledge
    assistant_knowledge_result = analyze_assistant_knowledge(dialogs, client)
    
    return {
        "profile": profile.replace("【User Profile】", "").strip(),
        "private": user_data.strip(),
        "assistant_knowledge": assistant_knowledge_result["assistant_knowledge"]
    }

def gpt_update_profile(old_profile, new_analysis, client):
    """
    Dynamically merges old and new profile data
    Args:
        old_profile: Previous profile text (structured)
        new_analysis: New analysis text (same format)
    Returns:
        Merged profile text with conflict resolution
    """
    prompt = f"""
# Profile Merge Task
Consolidate these profiles while:
- Preserving all valid observations
- Resolving conflicts
- Adding new dimensions

## Current Profile
{old_profile}

## New Data
{new_analysis}

## Rules
1. Keep ALL verified traits from both
2. Resolve conflicts by:
   a) New explicit evidence > old assumptions
   b) Mark as Neutral if contradictory
3. Add new dimensions from new data
4. Maintain EXACT original format

Output ONLY the merged profile (no commentary):
The generated content should not exceed 1500 words
"""

    messages = [
        {
            "role": "system",
            "content": """You are a profile integration system. Your rules:
1. NEVER discard verified information
2. Conflict resolution hierarchy:
   Explicit statement > Implied trait > Assumption
3. Add timestamps when traits change:
   (Updated: [date]) for modified traits
4. Preserve the 4-category structure"""
        },
        {"role": "user", "content": prompt}
    ]

    print("Updating user profile dynamically...")
    return gpt_generate_answer(prompt, messages, client)

def gpt_extract_theme(answer_text, client):
    prompt = f"请从以下回答中提取主题总结，并以【主题提取】：开头输出：\n{answer_text}\n"
    messages = [
        {"role": "system", "content": "You are an expert in extracting conversation topics."},
        {"role": "user", "content": prompt}
    ]
    print("调用 GPT 提取主题总结...")
    return gpt_generate_answer(prompt, messages, client)

def llm_extract_keywords(text, client):
    prompt = "Please extract the keywords of the conversation topic from the following dialogue, separated by commas, and do not exceed three:\n" + text
    messages = [
        {"role": "system", "content": "You are a keyword extraction expert. Please extract the keywords of the conversation topic."},
        {"role": "user", "content": prompt}
    ]
    print("调用 GPT 提取关键词...")
    keywords_text =gpt_generate_answer(prompt, messages, client)
    keywords = [w.strip() for w in keywords_text.split(",") if w.strip()]
    return set(keywords)

def compute_time_decay(session_timestamp, current_timestamp, tau=3600):
    from datetime import datetime
    fmt = "%Y-%m-%d %H:%M:%S"
    t1 = datetime.strptime(session_timestamp, fmt)
    t2 = datetime.strptime(current_timestamp, fmt)
    delta = (t2 - t1).total_seconds()
    return np.exp(-delta/tau)
