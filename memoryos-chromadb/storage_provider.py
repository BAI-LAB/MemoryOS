import chromadb
import json
import os
from typing import List, Dict, Any, Optional
from collections import defaultdict, deque
try:
    from .utils import get_timestamp, generate_id
except ImportError:
    from utils import get_timestamp, generate_id

class ChromaStorageProvider:
    """
    ChromaDB-based storage provider that replaces JSON files and Faiss indexes.
    Maintains all the original functionality while providing better performance and scalability.
    """
    
    def __init__(self, path: str, user_id: str, assistant_id: str, distance_function: str = "cosine"):
        self.path = path
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.distance_function = distance_function
        
        os.makedirs(path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=path)
        
        collection_metadata = {"hnsw:space": self.distance_function}

        def _get_or_create_collection_with_space(name: str):
            return self.client.get_or_create_collection(name=name, metadata=collection_metadata)

        self.mid_term_collection = _get_or_create_collection_with_space(f"mid_term_memory_user_{user_id}")
        self.user_knowledge_collection = _get_or_create_collection_with_space(f"user_knowledge_{user_id}")
        self.assistant_knowledge_collection = _get_or_create_collection_with_space(f"assistant_knowledge_{assistant_id}")
        
        self.metadata_file = os.path.join(path, f"metadata_{user_id}_{assistant_id}.json")
        self.metadata = self._load_metadata()
    
    def _distance_to_similarity(self, distance: float) -> float:
        if self.distance_function == "cosine":
            return max(0.0, 1.0 - distance)
        elif self.distance_function == "l2":
            import math
            return math.exp(-distance)
        elif self.distance_function == "ip":
            return (distance + 1) / 2
        else:
            return max(0.0, 1.0 - distance)
        
    def _safe_str(self, value) -> str:
        if value is None: return ""
        if isinstance(value, (list, dict)): return json.dumps(value)
        return str(value)
    
    def _safe_metadata(self, metadata_dict: dict) -> dict:
        return {k: self._safe_str(v) for k, v in metadata_dict.items() if v is not None}
        
    def _load_metadata(self) -> dict:
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"ChromaStorageProvider: Could not load metadata: {e}")
        
        return {
            "mid_term_sessions": {},
            "access_frequency": {},
            "heap_state": [],
            "short_term_memory": [],
            "user_profiles": {},
            "update_times": {}
        }
    
    def save_all_metadata(self):
        """Saves all metadata to the JSON file. This should be the ONLY method that writes to the file."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"ChromaStorageProvider: Error saving metadata: {e}")

    # ==================== SHORT TERM MEMORY ====================
    
    def add_short_term_memory(self, qa_pair: dict):
        if 'timestamp' not in qa_pair or not qa_pair['timestamp']:
            qa_pair["timestamp"] = get_timestamp()
        
        self.metadata.setdefault("short_term_memory", []).append(qa_pair)
        
    def get_short_term_memory(self, max_capacity: int) -> deque:
        return deque(self.metadata.get("short_term_memory", []), maxlen=max_capacity)
    
    def pop_oldest_short_term(self) -> Optional[dict]:
        memory_list = self.metadata.get("short_term_memory", [])
        return memory_list.pop(0) if memory_list else None
    
    def is_short_term_full(self, max_capacity: int) -> bool:
        return len(self.metadata.get("short_term_memory", [])) >= max_capacity

    # ==================== MID TERM MEMORY ====================
    
    def add_mid_term_session(self, session_data: dict, pages: List[dict]):
        session_id = session_data["id"]
        
        self.metadata.setdefault("mid_term_sessions", {})
        
        session_backup = {
            "id": session_id,
            "summary": session_data["summary"],
            "summary_keywords": session_data["summary_keywords"],
            "L_interaction": session_data["L_interaction"],
            "R_recency": session_data["R_recency"],
            "N_visit": session_data["N_visit"],
            "H_segment": session_data["H_segment"],
            "timestamp": session_data["timestamp"],
            "last_visit_time": session_data["last_visit_time"],
            "access_count_lfu": session_data["access_count_lfu"],
            "page_count": len(pages),
            "pages_backup": []
        }
        
        for page in pages:
            session_backup["pages_backup"].append({
                "page_id": page["page_id"],
                "user_input": page["user_input"],
                "agent_response": page["agent_response"],
                "timestamp": page["timestamp"],
                "preloaded": page.get("preloaded", False),
                "analyzed": page.get("analyzed", False),
                "pre_page": page.get("pre_page", ""),
                "next_page": page.get("next_page", ""),
                "meta_info": page.get("meta_info", ""),
                "page_keywords": page.get("page_keywords", [])
            })
        
        self.metadata["mid_term_sessions"][session_id] = session_backup
        
        session_metadata = self._safe_metadata({
            "type": "session_summary",
            "session_id": session_id,
            "summary": session_data["summary"],
            "timestamp": session_data["timestamp"],
            "user_id": self.user_id
        })
        
        if "summary_embedding" in session_data and session_data["summary_embedding"]:
            self.mid_term_collection.add(
                embeddings=[session_data["summary_embedding"]],
                metadatas=[session_metadata],
                ids=[f"session_{session_id}"]
            )
        
        if pages:
            # First, filter pages to only include those with embeddings.
            pages_with_embeddings = [p for p in pages if "page_embedding" in p and p["page_embedding"]]

            if pages_with_embeddings:
                page_embeddings = [p["page_embedding"] for p in pages_with_embeddings]
                page_metadatas = [self._safe_metadata({
                    "type": "page", "session_id": session_id, "page_id": p["page_id"],
                    "user_input": p["user_input"], "agent_response": p["agent_response"],
                    "timestamp": p["timestamp"], "user_id": self.user_id
                }) for p in pages_with_embeddings]
                page_ids = [f"page_{p['page_id']}" for p in pages_with_embeddings]
                
                # Ensure all lists have the same length before calling ChromaDB.
                if not (len(page_embeddings) == len(page_metadatas) == len(page_ids)):
                    print(f"CRITICAL ERROR in add_mid_term_session: Mismatched list lengths. "
                          f"Embeddings: {len(page_embeddings)}, Metadatas: {len(page_metadatas)}, IDs: {len(page_ids)}")
                    return 

                self.mid_term_collection.add(embeddings=page_embeddings, metadatas=page_metadatas, ids=page_ids)
    
    def search_mid_term_sessions(self, query_embedding: List[float], top_k: int = 5) -> List[dict]:
        try:
            results = self.mid_term_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 2, 100),
                where={"$and": [{"type": {"$eq": "session_summary"}}, {"user_id": {"$eq": self.user_id}}]}
            )
            sessions = []
            if results and results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i]
                    session_id = metadata['session_id']
                    session_meta = self.metadata.get("mid_term_sessions", {}).get(session_id)
                    if session_meta:
                        sessions.append({
                            "session_id": session_id,
                            "session_summary": metadata['summary'],
                            "session_relevance_score": self._distance_to_similarity(results['distances'][0][i]),
                            "session_metadata": session_meta
                        })
            return sorted(sessions, key=lambda x: x['session_relevance_score'], reverse=True)[:top_k]
        except Exception as e:
            print(f"ChromaStorageProvider: Error searching sessions: {e}")
            return []

    def get_pages_from_json_backup(self, session_id: str) -> List[dict]:
        return self.metadata.get("mid_term_sessions", {}).get(session_id, {}).get("pages_backup", [])

    def search_mid_term_pages(self, query_embedding: List[float], session_ids: List[str], top_k: int = 10) -> List[dict]:
        if not session_ids: return []
        try:
            results = self.mid_term_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 5, 200),
                where={"$and": [
                    {"type": {"$eq": "page"}},
                    {"session_id": {"$in": session_ids}},
                    {"user_id": {"$eq": self.user_id}}
                ]}
            )
            pages = []
            if results and results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i]
                    pages.append({
                        "page_id": metadata['page_id'],
                        "session_id": metadata['session_id'],
                        "user_input": metadata.get('user_input', ''),
                        "agent_response": metadata.get('agent_response', ''),
                        "relevance_score": self._distance_to_similarity(results['distances'][0][i])
                    })
            return sorted(pages, key=lambda x: x['relevance_score'], reverse=True)[:top_k]
        except Exception as e:
            print(f"ChromaStorageProvider: Error searching pages: {e}")
            return []

    def get_mid_term_sessions(self) -> dict:
        return self.metadata.get("mid_term_sessions", {})

    def update_mid_term_session_metadata(self, session_id: str, updates: dict):
        if session_id in self.metadata.get("mid_term_sessions", {}):
            self.metadata["mid_term_sessions"][session_id].update(updates)

    def delete_mid_term_session(self, session_id: str):
        if session_id in self.metadata.get("mid_term_sessions", {}):
            del self.metadata["mid_term_sessions"][session_id]
        if session_id in self.metadata.get("access_frequency", {}):
            del self.metadata["access_frequency"][session_id]
        try:
            self.mid_term_collection.delete(where={"session_id": session_id})
        except Exception as e:
            print(f"ChromaStorageProvider: Error deleting session {session_id} from ChromaDB: {e}")

    def get_page_by_id(self, page_id: str) -> Optional[dict]:
        try:
            result = self.mid_term_collection.get(ids=[f"page_{page_id}"], include=["metadatas"])
            if result and result['ids']:
                return result['metadatas'][0]
        except Exception as e:
            print(f"ChromaStorageProvider: Error getting page by id: {e}")
        return None

    def get_page_full_info(self, page_id: str, session_id: str) -> Optional[dict]:
        """获取页面的完整信息，包括meta_info等字段"""
        pages_backup = self.get_pages_from_json_backup(session_id)
        for page in pages_backup:
            if page.get("page_id") == page_id:
                return page
        return None

    def update_page_connections(self, page_id: str, updates: dict):
        page_chroma_info = self.get_page_by_id(page_id)
        if not page_chroma_info: return
        session_id = page_chroma_info.get("session_id")
        if not (session_id and session_id in self.metadata.get("mid_term_sessions", {})): return
        
        for page_backup in self.metadata["mid_term_sessions"][session_id].get("pages_backup", []):
            if page_backup.get("page_id") == page_id:
                page_backup.update(updates)
                break
    
    def get_access_frequency(self) -> defaultdict:
        return defaultdict(int, self.metadata.get("access_frequency", {}))
        
    def update_access_frequency(self, session_id: str, count: int):
        self.metadata.setdefault("access_frequency", {})[session_id] = count
        
    def get_heap_state(self) -> List:
        # JSON has no tuples, so entries saved as (-heat, session_id) come back
        # as lists. heapq cannot compare a list with the tuples pushed later.
        return [tuple(item) for item in self.metadata.get("heap_state", [])]
        
    def save_heap_state(self, heap: List):
        self.metadata["heap_state"] = heap

    # ==================== LONG TERM MEMORY (KNOWLEDGE) ====================

    def _add_knowledge(self, collection, text: str, embedding: List[float], metadata: dict):
        doc_id = generate_id(text)
        collection.add(embeddings=[embedding], metadatas=[self._safe_metadata(metadata)], ids=[doc_id])

    def add_user_knowledge(self, text: str, embedding: List[float]):
        metadata = {"type": "user_knowledge", "timestamp": get_timestamp(), "text": text}
        self._add_knowledge(self.user_knowledge_collection, text, embedding, metadata)

    def add_assistant_knowledge(self, text: str, embedding: List[float]):
        metadata = {"type": "assistant_knowledge", "timestamp": get_timestamp(), "text": text}
        self._add_knowledge(self.assistant_knowledge_collection, text, embedding, metadata)
    
    def _search_knowledge(self, collection, query_embedding: List[float], top_k: int) -> List[dict]:
        try:
            results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
            docs = []
            if results and results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    docs.append({
                        "id": doc_id,
                        "text": results['metadatas'][0][i]['text'],
                        "similarity": self._distance_to_similarity(results['distances'][0][i]),
                        "timestamp": results['metadatas'][0][i].get('timestamp')
                    })
            return docs
        except Exception:
            return []

    def search_user_knowledge(self, query_embedding: List[float], top_k: int = 5) -> List[dict]:
        return self._search_knowledge(self.user_knowledge_collection, query_embedding, top_k)

    def search_assistant_knowledge(self, query_embedding: List[float], top_k: int = 5) -> List[dict]:
        return self._search_knowledge(self.assistant_knowledge_collection, query_embedding, top_k)

    def _get_all_knowledge(self, collection) -> List[dict]:
        try:
            results = collection.get(include=["metadatas"])
            return [{"id": item_id, "text": meta.get("text", "")} for item_id, meta in zip(results['ids'], results['metadatas'])]
        except Exception:
            return []

    def get_all_user_knowledge(self) -> List[dict]:
        return self._get_all_knowledge(self.user_knowledge_collection)

    def get_all_assistant_knowledge(self) -> List[dict]:
        return self._get_all_knowledge(self.assistant_knowledge_collection)

    def enforce_knowledge_capacity(self, knowledge_type: str, max_capacity: int):
        collection = self.user_knowledge_collection if knowledge_type == "user" else self.assistant_knowledge_collection
        current_count = collection.count()
        if current_count > max_capacity:
            results = collection.get(include=["metadatas"], offset=0, limit=current_count)
            # Zip ids and metadatas together to sort them correctly.
            all_items = list(zip(results['ids'], results['metadatas']))
            # Sort by timestamp found in metadata.
            sorted_items = sorted(all_items, key=lambda x: x[1].get('timestamp', '0'))
            # Get the IDs of the items to delete.
            ids_to_delete = [item[0] for item in sorted_items[:current_count - max_capacity]]
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)

    # ==================== USER PROFILES & UPDATE TIMES ====================
    
    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]):
        self.metadata.setdefault("user_profiles", {})[user_id] = profile_data

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self.metadata.get("user_profiles", {}).get(user_id)

    def record_update_time(self, update_type: str, timestamp: str):
        """Records the timestamp of a specific update type ('profile' or 'knowledge')."""
        self.metadata.setdefault("update_times", {})[update_type] = timestamp

    def get_last_update_time(self, update_type: str) -> Optional[str]:
        """Gets the last update timestamp for a specific type."""
        return self.metadata.get("update_times", {}).get(update_type) 