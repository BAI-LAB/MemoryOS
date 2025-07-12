import time
import uuid
import openai
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
import prompts # Assuming prompts.py is in the same directory
from openai import OpenAI
# ---- OpenAI Client ----
class OpenAIClient:
    def __init__(self, api_key, base_url=None):
        self.api_key = api_key
        self.base_url = base_url if base_url else "https://api.openai.com/v1"
        # The openai library looks for OPENAI_API_KEY and OPENAI_BASE_URL env vars by default
        # or they can be passed directly to the client.
        # For simplicity and explicit control, we'll pass them to the client constructor.
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat_completion(self, model, messages, temperature=0.7, max_tokens=2000):
        print(f"Calling OpenAI API. Model: {model}")
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            # Fallback or error handling
            return "Error: Could not get response from LLM."


# ---- Basic Utilities ----
def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def generate_id(prefix="id"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def ensure_directory_exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ---- Embedding Utilities ----
_model_cache = {}

def get_embedding(text, model_name="all-MiniLM-L6-v2"):
    if model_name not in _model_cache:
        print(f"Loading sentence transformer model: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name)
    model = _model_cache[model_name]
    embedding = model.encode([text], convert_to_numpy=True)[0]
    return embedding

def normalize_vector(vec):
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# ---- Time Decay Function ----
def compute_time_decay(event_timestamp_str, current_timestamp_str, tau_hours=24):

    from datetime import datetime
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        t_event = datetime.strptime(event_timestamp_str, fmt)
        t_current = datetime.strptime(current_timestamp_str, fmt)
        delta_hours = (t_current - t_event).total_seconds() / 3600.0
        return np.exp(-delta_hours / tau_hours)
    except ValueError: # Handle cases where timestamp might be invalid
        return 0.1 # Default low recency


# ---- LLM-based Utility Functions ----

def gpt_summarize_dialogs(dialogs, client: OpenAIClient, model="gpt-4o-mini"):
    model=os.environ.get("llm_model") or model
    dialog_text = "\n".join([f"User: {d.get('user_input','')} Assistant: {d.get('agent_response','')}" for d in dialogs])
    messages = [
        {"role": "system", "content": prompts.SUMMARIZE_DIALOGS_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.SUMMARIZE_DIALOGS_USER_PROMPT.format(dialog_text=dialog_text)}
    ]
    print("Calling LLM to generate topic summary...")
    return client.chat_completion(model=model, messages=messages)

def gpt_generate_multi_summary(text, client: OpenAIClient, model="gpt-4o-mini"):
    model=os.environ.get("llm_model") or model
    messages = [
        {"role": "system", "content": prompts.MULTI_SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.MULTI_SUMMARY_USER_PROMPT.format(text=text)}
    ]
    print("Calling LLM to generate multi-topic summary...")
    response_text = client.chat_completion(model=model, messages=messages)
    try:
        summaries = json.loads(response_text)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse multi-summary JSON: {response_text}")
        summaries = [] # Return empty list or a default structure
    return {"input": text, "summaries": summaries}


def gpt_user_profile_analysis(dialogs, client: OpenAIClient, model="gpt-4o-mini", known_user_traits="None"):
    """Analyze user personality profile from dialogs"""
    model=os.environ.get("llm_model") or model
    conversation = "\n".join([f"User: {d.get('user_input','')} (Timestamp: {d.get('timestamp', '')})\nAssistant: {d.get('agent_response','')} (Timestamp: {d.get('timestamp', '')})" for d in dialogs])
    messages = [
        {"role": "system", "content": prompts.PERSONALITY_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.PERSONALITY_ANALYSIS_USER_PROMPT.format(
            conversation=conversation,
            known_user_traits=known_user_traits
        )}
    ]
    print("Calling LLM for user profile analysis...")
    result_text = client.chat_completion(model=model, messages=messages)
    return result_text.strip() if result_text else "None"


def gpt_knowledge_extraction(dialogs, client: OpenAIClient, model="gpt-4o-mini"):
    """Extract user private data and assistant knowledge from dialogs"""
    model=os.environ.get("llm_model") or model
    conversation = "\n".join([f"User: {d.get('user_input','')} (Timestamp: {d.get('timestamp', '')})\nAssistant: {d.get('agent_response','')} (Timestamp: {d.get('timestamp', '')})" for d in dialogs])
    messages = [
        {"role": "system", "content": prompts.KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.KNOWLEDGE_EXTRACTION_USER_PROMPT.format(
            conversation=conversation
        )}
    ]
    print("Calling LLM for knowledge extraction...")
    result_text = client.chat_completion(model=model, messages=messages)
    
    private_data = "None"
    assistant_knowledge = "None"

    try:
        if "【User Private Data】" in result_text:
            private_data_start = result_text.find("【User Private Data】") + len("【User Private Data】")
            if "【Assistant Knowledge】" in result_text:
                private_data_end = result_text.find("【Assistant Knowledge】")
                private_data = result_text[private_data_start:private_data_end].strip()
                
                assistant_knowledge_start = result_text.find("【Assistant Knowledge】") + len("【Assistant Knowledge】")
                assistant_knowledge = result_text[assistant_knowledge_start:].strip()
            else:
                private_data = result_text[private_data_start:].strip()
        elif "【Assistant Knowledge】" in result_text:
             assistant_knowledge_start = result_text.find("【Assistant Knowledge】") + len("【Assistant Knowledge】")
             assistant_knowledge = result_text[assistant_knowledge_start:].strip()

    except Exception as e:
        print(f"Error parsing knowledge extraction: {e}. Raw result: {result_text}")

    return {
        "private": private_data if private_data else "None", 
        "assistant_knowledge": assistant_knowledge if assistant_knowledge else "None"
    }


# Keep the old function for backward compatibility, but mark as deprecated
def gpt_personality_analysis(dialogs, client: OpenAIClient, model="gpt-4o-mini", known_user_traits="None"):
    """
    DEPRECATED: Use gpt_user_profile_analysis and gpt_knowledge_extraction instead.
    This function is kept for backward compatibility only.
    """
    model=os.environ.get("llm_model") or model
    # Call the new functions
    profile = gpt_user_profile_analysis(dialogs, client, model, known_user_traits)
    knowledge_data = gpt_knowledge_extraction(dialogs, client, model)
    
    return {
        "profile": profile,
        "private": knowledge_data["private"],
        "assistant_knowledge": knowledge_data["assistant_knowledge"]
    }


def gpt_update_profile(old_profile, new_analysis, client: OpenAIClient, model="gpt-4o-mini"):
    model=os.environ.get("llm_model") or model
    messages = [
        {"role": "system", "content": prompts.UPDATE_PROFILE_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.UPDATE_PROFILE_USER_PROMPT.format(old_profile=old_profile, new_analysis=new_analysis)}
    ]
    print("Calling LLM to update user profile...")
    return client.chat_completion(model=model, messages=messages)

def gpt_extract_theme(answer_text, client: OpenAIClient, model="gpt-4o-mini"):
    model=os.environ.get("llm_model") or model
    messages = [
        {"role": "system", "content": prompts.EXTRACT_THEME_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.EXTRACT_THEME_USER_PROMPT.format(answer_text=answer_text)}
    ]
    print("Calling LLM to extract theme...")
    return client.chat_completion(model=model, messages=messages)

def llm_extract_keywords(text, client: OpenAIClient, model="gpt-4o-mini"):
    model=os.environ.get("llm_model") or model

    messages = [
        {"role": "system", "content": prompts.EXTRACT_KEYWORDS_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.EXTRACT_KEYWORDS_USER_PROMPT.format(text=text)}
    ]
    print("Calling LLM to extract keywords...")
    response = client.chat_completion(model=model, messages=messages)
    return [kw.strip() for kw in response.split(',') if kw.strip()]

# ---- Functions from dynamic_update.py (to be used by Updater class) ----
def check_conversation_continuity(previous_page, current_page, client: OpenAIClient, model="gpt-4o-mini"):
    model=os.environ.get("llm_model") or model
    prev_user = previous_page.get("user_input", "") if previous_page else ""
    prev_agent = previous_page.get("agent_response", "") if previous_page else ""
    
    user_prompt = prompts.CONTINUITY_CHECK_USER_PROMPT.format(
        prev_user=prev_user,
        prev_agent=prev_agent,
        curr_user=current_page.get("user_input", ""),
        curr_agent=current_page.get("agent_response", "")
    )
    messages = [
        {"role": "system", "content": prompts.CONTINUITY_CHECK_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    response = client.chat_completion(model=model, messages=messages, temperature=0.0, max_tokens=10)
    return response.strip().lower() == "true"

def generate_page_meta_info(last_page_meta, current_page, client: OpenAIClient, model="gpt-4o-mini"):
    model=os.environ.get("llm_model") or model
    current_conversation = f"User: {current_page.get('user_input', '')}\nAssistant: {current_page.get('agent_response', '')}"
    user_prompt = prompts.META_INFO_USER_PROMPT.format(
        last_meta=last_page_meta if last_page_meta else "None",
        new_dialogue=current_conversation
    )
    messages = [
        {"role": "system", "content": prompts.META_INFO_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    return client.chat_completion(model=model, messages=messages, temperature=0.3, max_tokens=100).strip() 
