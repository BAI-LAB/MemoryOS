import time
import uuid
import openai
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
import inspect
from functools import wraps
try:
    from . import prompts # 尝试相对导入
except ImportError:
    import prompts # 回退到绝对导入
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def clean_reasoning_model_output(text):
    """
    清理推理模型输出中的<think>标签
    """
    if not text:
        return text
    
    import re
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

# ---- OpenAI Client ----
class OpenAIClient:
    def __init__(self, api_key, base_url=None, max_workers=5):
        self.api_key = api_key
        self.base_url = base_url if base_url else "https://api.openai.com/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()

    def chat_completion(self, model, messages, temperature=0.7, max_tokens=2000):
        max_retries = 3
        for attempt in range(max_retries):
            print(f"Calling OpenAI API. Model: {model} (attempt {attempt + 1}/{max_retries})")
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                raw_content = response.choices[0].message.content.strip()
                cleaned_content = clean_reasoning_model_output(raw_content)
                return cleaned_content
            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(code in error_str for code in ["429", "502", "503", "timeout", "connection"])
                if is_retryable and attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Error calling OpenAI API: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Error calling OpenAI API: {e}")
                    return "Error: Could not get response from LLM."
        return "Error: Could not get response from LLM."

    def chat_completion_async(self, model, messages, temperature=0.7, max_tokens=2000):
        return self.executor.submit(self.chat_completion, model, messages, temperature, max_tokens)

    def batch_chat_completion(self, requests):
        futures = [self.chat_completion_async(**req) for req in requests]
        results = [future.result() for future in as_completed(futures)]
        return results

    def shutdown(self):
        self.executor.shutdown(wait=True)

# ---- Parallel Processing Utilities ----
def run_parallel_tasks(tasks, max_workers=3):
    """
    并行执行任务列表
    tasks: List of callable functions
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task) for task in tasks]
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error in parallel task: {e}")
                results.append(None)
        return results

# ---- Basic Utilities ----
def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def generate_id(prefix="id"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def ensure_directory_exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ---- Embedding Utilities ----
_model_cache = {}
_embedding_cache = {}

def _get_valid_kwargs(func, kwargs):
    try:
        sig = inspect.signature(func)
        param_keys = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in param_keys}
    except (ValueError, TypeError):
        return kwargs

def get_embedding(text, model_name="all-MiniLM-L6-v2", use_cache=True, **kwargs):
    model_config_key = json.dumps({"model_name": model_name, **kwargs}, sort_keys=True)
    
    if use_cache:
        cache_key = f"{model_config_key}::{hash(text)}"
        if cache_key in _embedding_cache:
            return _embedding_cache[cache_key]
    
    model_init_key = json.dumps({"model_name": model_name, **{k:v for k,v in kwargs.items() if k not in ['batch_size', 'max_length']}}, sort_keys=True)
    if model_init_key not in _model_cache:
        print(f"Loading model: {model_name}...")
        if 'bge-m3' in model_name.lower():
            try:
                from FlagEmbedding import BGEM3FlagModel
                init_kwargs = _get_valid_kwargs(BGEM3FlagModel.__init__, kwargs)
                _model_cache[model_init_key] = BGEM3FlagModel(model_name, **init_kwargs)
            except ImportError:
                raise ImportError("Please install FlagEmbedding: 'pip install -U FlagEmbedding' to use bge-m3 model.")
        else:
            from sentence_transformers import SentenceTransformer
            init_kwargs = _get_valid_kwargs(SentenceTransformer.__init__, kwargs)
            _model_cache[model_init_key] = SentenceTransformer(model_name, **init_kwargs)
            
    model = _model_cache[model_init_key]
    
    embedding = None
    if 'bge-m3' in model_name.lower():
        encode_kwargs = _get_valid_kwargs(model.encode, kwargs)
        result = model.encode([text], **encode_kwargs)
        embedding = result['dense_vecs'][0]
    else:
        encode_kwargs = _get_valid_kwargs(model.encode, kwargs)
        embedding = model.encode([text], **encode_kwargs)[0]

    if use_cache:
        cache_key = f"{model_config_key}::{hash(text)}"
        _embedding_cache[cache_key] = embedding
    
    return embedding

def normalize_vector(vec):
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

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
    dialog_text = "\n".join([f"User: {d.get('user_input','')} Assistant: {d.get('agent_response','')}" for d in dialogs])
    messages = [
        {"role": "system", "content": prompts.SUMMARIZE_DIALOGS_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.SUMMARIZE_DIALOGS_USER_PROMPT.format(dialog_text=dialog_text)}
    ]
    print("Calling LLM to generate topic summary...")
    return client.chat_completion(model=model, messages=messages)

def gpt_generate_multi_summary(text, client: OpenAIClient, model="gpt-4o-mini"):
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
        summaries = []
    return {"input": text, "summaries": summaries}

def extract_keywords_from_multi_summary(text, client: OpenAIClient, model="gpt-4o-mini"):
    """
    Extract keywords using multi-summary analysis instead of separate keyword extraction.
    This is more efficient as the multi-summary already includes keywords for each theme.
    """
    multi_summary_result = gpt_generate_multi_summary(text, client, model)
    all_keywords = []
    
    if multi_summary_result and multi_summary_result.get("summaries"):
        for summary_item in multi_summary_result["summaries"]:
            keywords = summary_item.get("keywords", [])
            all_keywords.extend(keywords)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in all_keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)
    
    return unique_keywords

def gpt_user_profile_analysis(conversation_str: str, client: OpenAIClient, model="gpt-4o-mini", existing_user_profile="None"):
    """
    Analyze and update user personality profile from a conversation string.
    """
    messages = [
        {"role": "system", "content": prompts.PERSONALITY_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.PERSONALITY_ANALYSIS_USER_PROMPT.format(
            conversation=conversation_str,
            existing_user_profile=existing_user_profile
        )}
    ]
    print("Calling LLM for user profile analysis and update...")
    result_text = client.chat_completion(model=model, messages=messages)
    try:
        return json.loads(result_text)
    except json.JSONDecodeError:
        print(f"Warning: User profile analysis did not return valid JSON. Content: {result_text}")
        return {"raw_text_profile": result_text}

def gpt_knowledge_extraction(conversation_str: str, client: OpenAIClient, model="gpt-4o-mini"):
    """Extract user private data and assistant knowledge from a conversation string"""
    messages = [
        {"role": "system", "content": prompts.KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.KNOWLEDGE_EXTRACTION_USER_PROMPT.format(
            conversation=conversation_str
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

def gpt_update_profile(old_profile, new_analysis, client: OpenAIClient, model="gpt-4o-mini"):
    messages = [
        {"role": "system", "content": prompts.UPDATE_PROFILE_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.UPDATE_PROFILE_USER_PROMPT.format(old_profile=old_profile, new_analysis=new_analysis)}
    ]
    return client.chat_completion(model=model, messages=messages)

def check_conversation_continuity(previous_page, current_page, client: OpenAIClient, model="gpt-4o-mini"):
    if not previous_page or not current_page:
        return False
    
    prompt = prompts.CONTINUITY_CHECK_USER_PROMPT.format(
        prev_user=previous_page.get('user_input', ''),
        prev_agent=previous_page.get('agent_response', ''),
        curr_user=current_page.get('user_input', ''),
        curr_agent=current_page.get('agent_response', '')
    )
    messages = [{"role": "system", "content": prompts.CONTINUITY_CHECK_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    
    response = client.chat_completion(model, messages, temperature=0.0)
    return response.lower() == 'true'

def generate_page_meta_info(last_page_meta, current_page, client: OpenAIClient, model="gpt-4o-mini"):
    new_dialogue = f"User: {current_page.get('user_input', '')}\nAssistant: {current_page.get('agent_response', '')}"
    
    prompt = prompts.META_INFO_USER_PROMPT.format(
        last_meta=last_page_meta or "This is the beginning of the conversation.",
        new_dialogue=new_dialogue
    )
    messages = [{"role": "system", "content": prompts.META_INFO_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    
    return client.chat_completion(model, messages, temperature=0.3) 