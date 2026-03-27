import time
import uuid
import numpy as np
import json
import os
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
    适配推理模型（如o1系列）的输出格式
    """
    if not text:
        return text
    
    import re
    # 移除<think>...</think>标签及其内容
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 清理可能产生的多余空白行
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
    # 移除开头和结尾的空白
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

# ---- OpenAI Client ----
class OpenAIClient:
    def __init__(self, api_key, base_url=None, max_workers=5,
                 embedding_api_key=None, embedding_base_url=None, embedding_model="text-embedding-3-small"):
        self.api_key = api_key
        self.base_url = base_url if base_url else "https://api.openai.com/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()

        # Embedding client (can be a separate provider)
        self.embedding_model = embedding_model
        _emb_key = embedding_api_key or self.api_key
        _emb_url = embedding_base_url or self.base_url
        self.embedding_client = OpenAI(api_key=_emb_key, base_url=_emb_url)
        self._embedding_cache = {}

    def chat_completion(self, model, messages, temperature=0.7, max_tokens=2000):
        print(f"Calling OpenAI API. Model: {model}")
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            raw_content = response.choices[0].message.content.strip()
            # 自动清理推理模型的<think>标签
            cleaned_content = clean_reasoning_model_output(raw_content)
            return cleaned_content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            # Fallback or error handling
            return "Error: Could not get response from LLM."

    def chat_completion_async(self, model, messages, temperature=0.7, max_tokens=2000):
        """异步版本的chat_completion"""
        return self.executor.submit(self.chat_completion, model, messages, temperature, max_tokens)

    def batch_chat_completion(self, requests):
        """
        并行处理多个LLM请求
        requests: List of dict with keys: model, messages, temperature, max_tokens
        """
        futures = []
        for req in requests:
            future = self.chat_completion_async(
                model=req.get("model", "gpt-4o-mini"),
                messages=req["messages"],
                temperature=req.get("temperature", 0.7),
                max_tokens=req.get("max_tokens", 2000)
            )
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error in batch completion: {e}")
                results.append("Error: Could not get response from LLM.")
        
        return results

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

    def shutdown(self):
        """关闭线程池"""
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
        summaries = [] # Return empty list or a default structure
    return {"input": text, "summaries": summaries}


def gpt_user_profile_analysis(dialogs, client: OpenAIClient, model="gpt-4o-mini", existing_user_profile="None"):
    """
    Analyze and update user personality profile from dialogs
    结合现有画像和新对话，直接输出更新后的完整画像
    """
    conversation = "\n".join([f"User: {d.get('user_input','')} (Timestamp: {d.get('timestamp', '')})\nAssistant: {d.get('agent_response','')} (Timestamp: {d.get('timestamp', '')})" for d in dialogs])
    messages = [
        {"role": "system", "content": prompts.PERSONALITY_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.PERSONALITY_ANALYSIS_USER_PROMPT.format(
            conversation=conversation,
            existing_user_profile=existing_user_profile
        )}
    ]
    print("Calling LLM for user profile analysis and update...")
    result_text = client.chat_completion(model=model, messages=messages)
    return result_text.strip() if result_text else "None"


def gpt_knowledge_extraction(dialogs, client: OpenAIClient, model="gpt-4o-mini"):
    """Extract user private data and assistant knowledge from dialogs"""
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
    # Call the new functions
    profile = gpt_user_profile_analysis(dialogs, client, model, known_user_traits)
    knowledge_data = gpt_knowledge_extraction(dialogs, client, model)
    
    return {
        "profile": profile,
        "private": knowledge_data["private"],
        "assistant_knowledge": knowledge_data["assistant_knowledge"]
    }


def gpt_update_profile(old_profile, new_analysis, client: OpenAIClient, model="gpt-4o-mini"):
    messages = [
        {"role": "system", "content": prompts.UPDATE_PROFILE_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.UPDATE_PROFILE_USER_PROMPT.format(old_profile=old_profile, new_analysis=new_analysis)}
    ]
    print("Calling LLM to update user profile...")
    return client.chat_completion(model=model, messages=messages)

def gpt_extract_theme(answer_text, client: OpenAIClient, model="gpt-4o-mini"):
    messages = [
        {"role": "system", "content": prompts.EXTRACT_THEME_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.EXTRACT_THEME_USER_PROMPT.format(answer_text=answer_text)}
    ]
    print("Calling LLM to extract theme...")
    return client.chat_completion(model=model, messages=messages)



# ---- Functions from dynamic_update.py (to be used by Updater class) ----
def check_conversation_continuity(previous_page, current_page, client: OpenAIClient, model="gpt-4o-mini"):
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