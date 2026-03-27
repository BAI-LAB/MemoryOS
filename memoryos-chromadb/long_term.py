import json
import numpy as np
from typing import Optional, Dict, Any

try:
    from .utils import get_timestamp, normalize_vector, OpenAIClient, gpt_user_profile_analysis, gpt_knowledge_extraction
    from .storage_provider import ChromaStorageProvider
except ImportError:
    from utils import get_timestamp, normalize_vector, OpenAIClient, gpt_user_profile_analysis, gpt_knowledge_extraction
    from storage_provider import ChromaStorageProvider

class LongTermMemory:
    def __init__(self,
                 storage_provider: ChromaStorageProvider,
                 llm_interface: OpenAIClient,
                 client: OpenAIClient = None,
                 knowledge_capacity=100,
                 llm_model: str = "gpt-4o-mini"):
        self.storage = storage_provider
        self.llm_interface = llm_interface
        self.client = client or llm_interface
        self.knowledge_capacity = knowledge_capacity
        self.llm_model = llm_model

    def update_user_profile(self, user_id: str, conversation_history: str) -> Optional[Dict[str, Any]]:
        """
        Generates a new user profile based on conversation history and updates it in storage.
        """
        existing_profile_str = json.dumps(self.get_user_profile(user_id) or {})
        
        updated_profile = gpt_user_profile_analysis(
            conversation_str=conversation_history,
            client=self.llm_interface,
            model=self.llm_model,  # 传递模型参数
            existing_user_profile=existing_profile_str
        )
        
        if updated_profile:
            self.storage.update_user_profile(user_id, updated_profile)
            print(f"LongTermMemory: Updated user profile for {user_id}.")
            return updated_profile
        return None

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self.storage.get_user_profile(user_id)

    def add_knowledge(self, knowledge_text: str, knowledge_type: str = "user"):
        """
        Adds a knowledge entry (for user or assistant) to ChromaDB.
        knowledge_type can be 'user' or 'assistant'.
        """
        print(f"DEBUG: add_knowledge received text: '{knowledge_text}'") # Debugging line
        if not knowledge_text or knowledge_text.strip().lower() in ["", "none", "- none", "- none."]:
            print(f"LongTermMemory: Empty {knowledge_type} knowledge received, not saving.")
            return
        
        vec = self.client.get_embedding(knowledge_text)
        vec = normalize_vector(vec).tolist()
        
        if knowledge_type == "user":
            self.storage.add_user_knowledge(knowledge_text, vec)
        else:
            self.storage.add_assistant_knowledge(knowledge_text, vec)
        
        self.storage.enforce_knowledge_capacity(knowledge_type, self.knowledge_capacity)

    def extract_knowledge_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Uses an LLM to extract structured knowledge from a block of text.
        """
        if not text.strip():
            return None
        return gpt_knowledge_extraction(
            conversation_str=text, 
            client=self.llm_interface,
            model=self.llm_model  # 传递模型参数
        )

    def get_user_knowledge(self) -> list:
        return self.storage.get_all_user_knowledge()

    def get_assistant_knowledge(self) -> list:
        return self.storage.get_all_assistant_knowledge()

    def search_knowledge(self, query: str, knowledge_type: str = "user", top_k=5) -> list:
        query_vec = self.client.get_embedding(query)
        query_vec = normalize_vector(query_vec).tolist()
        
        if knowledge_type == "user":
            results = self.storage.search_user_knowledge(query_vec, top_k=top_k)
        else:
            results = self.storage.search_assistant_knowledge(query_vec, top_k=top_k)
        
        print(f"LongTermMemory: Searched {knowledge_type} knowledge for '{query[:30]}...'. Found {len(results)} matches.")
        return results