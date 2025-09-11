import json
import numpy as np
import faiss
from collections import deque
try:
    from .utils import get_timestamp, get_embedding, normalize_vector, ensure_directory_exists
except ImportError:
    from utils import get_timestamp, get_embedding, normalize_vector, ensure_directory_exists

class LongTermMemory:
    def __init__(self, file_path, knowledge_capacity=100, embedding_model_name: str = "all-MiniLM-L6-v2", embedding_model_kwargs: dict = None):
        self.file_path = file_path
        ensure_directory_exists(self.file_path)
        self.knowledge_capacity = knowledge_capacity
        self.user_profiles = {} # {user_id: {data: "profile_string", "last_updated": "timestamp"}}
        # Use deques for knowledge bases to easily manage capacity
        self.knowledge_base = deque(maxlen=self.knowledge_capacity) # For general/user private knowledge
        self.assistant_knowledge = deque(maxlen=self.knowledge_capacity) # For assistant specific knowledge

        self.embedding_model_name = embedding_model_name
        self.embedding_model_kwargs = embedding_model_kwargs if embedding_model_kwargs is not None else {}
        self.load()

    def update_user_profile(self, user_id, new_data, merge=True):
        if merge and user_id in self.user_profiles and self.user_profiles[user_id].get("data"): # Check if data exists
            current_data = self.user_profiles[user_id]["data"]
            if isinstance(current_data, str) and isinstance(new_data, str):
                updated_data = f"{current_data}\n\n--- Updated on {get_timestamp()} ---\n{new_data}"
            else: # Fallback to overwrite if types are not strings or for more complex merge
                updated_data = new_data
        else:
            # If merge=False or no existing data, replace with new data
            updated_data = new_data
        
        self.user_profiles[user_id] = {
            "data": updated_data,
            "last_updated": get_timestamp()
        }
        print(f"LongTermMemory: Updated user profile for {user_id} (merge={merge}).")
        self.save()

    def get_raw_user_profile(self, user_id):
        return self.user_profiles.get(user_id, {}).get("data", "None") # Return "None" string if not found

    def get_user_profile_data(self, user_id):
        return self.user_profiles.get(user_id, {})

    def add_knowledge_entry(self, knowledge_text, knowledge_deque: deque, type_name="knowledge"):
        if not knowledge_text or knowledge_text.strip().lower() in ["", "none", "- none", "- none."]:
            print(f"LongTermMemory: Empty {type_name} received, not saving.")
            return
        
        # If deque is full, the oldest item is automatically removed when appending.
        vec = get_embedding(
            knowledge_text, 
            model_name=self.embedding_model_name, 
            **self.embedding_model_kwargs
        )
        vec = normalize_vector(vec).tolist()
        entry = {
            "knowledge": knowledge_text,
            "timestamp": get_timestamp(),
            "knowledge_embedding": vec
        }
        knowledge_deque.append(entry)
        print(f"LongTermMemory: Added {type_name}. Current count: {len(knowledge_deque)}.")
        self.save()

    def add_user_knowledge(self, knowledge_text):
        self.add_knowledge_entry(knowledge_text, self.knowledge_base, "user knowledge")

    def add_assistant_knowledge(self, knowledge_text):
        self.add_knowledge_entry(knowledge_text, self.assistant_knowledge, "assistant knowledge")

    def get_user_knowledge(self):
        return list(self.knowledge_base)

    def get_assistant_knowledge(self):
        return list(self.assistant_knowledge)

    def _search_knowledge_deque(self, query, knowledge_deque: deque, threshold=0.1, top_k=5):
        if not knowledge_deque:
            return []
        
        query_vec = get_embedding(
            query, 
            model_name=self.embedding_model_name, 
            **self.embedding_model_kwargs
        )
        query_vec = normalize_vector(query_vec)
        
        embeddings = []
        valid_entries = []
        for entry in knowledge_deque:
            if "knowledge_embedding" in entry and entry["knowledge_embedding"]:
                embeddings.append(np.array(entry["knowledge_embedding"], dtype=np.float32))
                valid_entries.append(entry)
            else:
                print(f"Warning: Entry without embedding found in knowledge_deque: {entry.get('knowledge','N/A')[:50]}")

        if not embeddings:
            return []
            
        embeddings_np = np.array(embeddings, dtype=np.float32)
        if embeddings_np.ndim == 1: # Single item case
            if embeddings_np.shape[0] == 0: return [] # Empty embeddings
            embeddings_np = embeddings_np.reshape(1, -1)
        
        if embeddings_np.shape[0] == 0: # No valid embeddings
            return []

        dim = embeddings_np.shape[1]
        index = faiss.IndexFlatIP(dim) # Using Inner Product for similarity
        index.add(embeddings_np)
        
        query_arr = np.array([query_vec], dtype=np.float32)
        distances, indices = index.search(query_arr, min(top_k, len(valid_entries))) # Search at most k or length of valid_entries
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1: # faiss returns -1 for no valid index
                similarity_score = float(distances[0][i]) # For IndexFlatIP, distance is the dot product (similarity)
                if similarity_score >= threshold:
                    results.append(valid_entries[idx]) # Add the original entry dict
        
        # Sort by similarity score descending before returning, as faiss might not guarantee order for IP
        results.sort(key=lambda x: float(np.dot(np.array(x["knowledge_embedding"], dtype=np.float32), query_vec)), reverse=True)
        return results

    def search_user_knowledge(self, query, threshold=0.1, top_k=5):
        results = self._search_knowledge_deque(query, self.knowledge_base, threshold, top_k)
        print(f"LongTermMemory: Searched user knowledge for '{query[:30]}...'. Found {len(results)} matches.")
        return results

    def search_assistant_knowledge(self, query, threshold=0.1, top_k=5):
        results = self._search_knowledge_deque(query, self.assistant_knowledge, threshold, top_k)
        print(f"LongTermMemory: Searched assistant knowledge for '{query[:30]}...'. Found {len(results)} matches.")
        return results

    def save(self):
        data = {
            "user_profiles": self.user_profiles,
            "knowledge_base": list(self.knowledge_base), # Convert deques to lists for JSON serialization
            "assistant_knowledge": list(self.assistant_knowledge)
        }
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Error saving LongTermMemory to {self.file_path}: {e}")

    def load(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.user_profiles = data.get("user_profiles", {})
                # Load into deques, respecting maxlen
                kb_data = data.get("knowledge_base", [])
                self.knowledge_base = deque(kb_data, maxlen=self.knowledge_capacity)
                
                ak_data = data.get("assistant_knowledge", [])
                self.assistant_knowledge = deque(ak_data, maxlen=self.knowledge_capacity)
                
            print(f"LongTermMemory: Loaded from {self.file_path}.")
        except FileNotFoundError:
            print(f"LongTermMemory: No history file found at {self.file_path}. Initializing new memory.")
        except json.JSONDecodeError:
            print(f"LongTermMemory: Error decoding JSON from {self.file_path}. Initializing new memory.")
        except Exception as e:
             print(f"LongTermMemory: An unexpected error occurred during load from {self.file_path}: {e}. Initializing new memory.") 