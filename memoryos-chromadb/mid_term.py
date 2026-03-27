import json
import numpy as np
from collections import defaultdict
import heapq
from datetime import datetime
from typing import Optional

try:
    from .utils import (
        get_timestamp, generate_id, normalize_vector,
        extract_keywords_from_multi_summary, compute_time_decay, ensure_directory_exists, OpenAIClient
    )
    from .storage_provider import ChromaStorageProvider
except ImportError:
    from utils import (
        get_timestamp, generate_id, normalize_vector,
        extract_keywords_from_multi_summary, compute_time_decay, ensure_directory_exists, OpenAIClient
    )
    from storage_provider import ChromaStorageProvider

# Heat computation constants (can be tuned or made configurable)
HEAT_ALPHA = 1.0
HEAT_BETA = 1.0
HEAT_GAMMA = 1
RECENCY_TAU_HOURS = 24 # For R_recency calculation in compute_segment_heat

def compute_segment_heat(session, alpha=HEAT_ALPHA, beta=HEAT_BETA, gamma=HEAT_GAMMA, tau_hours=RECENCY_TAU_HOURS):
    N_visit = session.get("N_visit", 0)
    L_interaction = session.get("L_interaction", 0)
    
    # Calculate recency based on last_visit_time
    R_recency = 1.0 # Default if no last_visit_time
    if session.get("last_visit_time"):
        R_recency = compute_time_decay(session["last_visit_time"], get_timestamp(), tau_hours)
    
    session["R_recency"] = R_recency # Update session's recency factor
    return alpha * N_visit + beta * L_interaction + gamma * R_recency

class MidTermMemory:
    def __init__(self,
                 storage_provider: ChromaStorageProvider,
                 user_id: str,
                 client: OpenAIClient,
                 max_capacity=2000,
                 llm_model: str = "gpt-4o-mini"):
        self.user_id = user_id
        self.client = client
        self.max_capacity = max_capacity
        self.storage = storage_provider
        self.llm_model = llm_model
        
        # Load sessions and other data from the shared storage provider's in-memory metadata
        self.sessions: dict = self.storage.get_mid_term_sessions()
        self.access_frequency: defaultdict[str, int] = self.storage.get_access_frequency()
        self.heap: list = self.storage.get_heap_state()
        
        # If heap is empty, rebuild it from loaded sessions
        if not self.heap and self.sessions:
            self.rebuild_heap()

    def get_page_by_id(self, page_id):
        return self.storage.get_page_by_id(page_id)

    def update_page_connections(self, prev_page_id, next_page_id):
        if prev_page_id:
            self.storage.update_page_connections(prev_page_id, {"next_page": next_page_id})
        if next_page_id:
            self.storage.update_page_connections(next_page_id, {"pre_page": prev_page_id})

    def evict_lfu(self):
        if not self.access_frequency or not self.sessions:
            return
        
        lfu_sid = min(self.access_frequency, key=lambda k: self.access_frequency[k])
        print(f"MidTermMemory: LFU eviction. Session {lfu_sid} has lowest access frequency.")
        
        if lfu_sid not in self.sessions:
            del self.access_frequency[lfu_sid] # Clean up access frequency if session already gone
            self.rebuild_heap()
            return
        
        # Remove from storage
        self.storage.delete_mid_term_session(lfu_sid)
        
        # Remove from local data structures
        session_to_delete = self.sessions.pop(lfu_sid)
        del self.access_frequency[lfu_sid]

        self.rebuild_heap()
        print(f"MidTermMemory: Evicted session {lfu_sid}.")

    def add_session(self, summary, details):
        session_id = generate_id("session")
        summary_vec = self.client.get_embedding(summary)
        summary_vec = normalize_vector(summary_vec).tolist()
        summary_keywords = list(extract_keywords_from_multi_summary(summary, client=self.client,model=self.llm_model))  
        
        processed_details = []
        for page_data in details:
            page_id = page_data.get("page_id", generate_id("page"))
            
            # 检查是否已有embedding，避免重复计算
            if "page_embedding" in page_data and page_data["page_embedding"]:
                print(f"MidTermMemory: Reusing existing embedding for page {page_id}")
                inp_vec = page_data["page_embedding"]
                # 确保embedding是normalized的
                if isinstance(inp_vec, list):
                    inp_vec_np = np.array(inp_vec, dtype=np.float32)
                    if np.linalg.norm(inp_vec_np) > 1.1 or np.linalg.norm(inp_vec_np) < 0.9:  # 检查是否需要重新normalize
                        inp_vec = normalize_vector(inp_vec_np).tolist()
            else:
                print(f"MidTermMemory: Computing new embedding for page {page_id}")
                full_text = f"User: {page_data.get('user_input','')} Assistant: {page_data.get('agent_response','')}"
                inp_vec = self.client.get_embedding(full_text)
                inp_vec = normalize_vector(inp_vec).tolist()
            
            # 检查是否已有keywords，避免重复计算
            if "page_keywords" in page_data and page_data["page_keywords"]:
                print(f"MidTermMemory: Reusing existing keywords for page {page_id}")
                page_keywords = page_data["page_keywords"]
            else:
                print(f"MidTermMemory: Computing new keywords for page {page_id}")
                full_text = f"User: {page_data.get('user_input','')} Assistant: {page_data.get('agent_response','')}"
                page_keywords = list(extract_keywords_from_multi_summary(full_text, client=self.client,model=self.llm_model))
            
            processed_page = {
                **page_data, # Carry over existing fields like user_input, agent_response, timestamp
                "page_id": page_id,
                "page_embedding": inp_vec,
                "page_keywords": page_keywords,
                "preloaded": page_data.get("preloaded", False), # Preserve if passed
                "analyzed": page_data.get("analyzed", False),   # Preserve if passed
                # pre_page, next_page, meta_info are handled by DynamicUpdater
            }
            processed_details.append(processed_page)
        
        current_ts = get_timestamp()
        session_obj = {
            "id": session_id,
            "summary": summary,
            "summary_keywords": summary_keywords,
            "summary_embedding": summary_vec,
            # Note: In ChromaDB refactor, "details" (pages) are stored in ChromaDB, not in session object
            "L_interaction": len(processed_details),
            "R_recency": 1.0, # Initial recency
            "N_visit": 0,
            "H_segment": 0.0, # Initial heat, will be computed
            "timestamp": current_ts, # Creation timestamp
            "last_visit_time": current_ts, # Also initial last_visit_time for recency calc
            "access_count_lfu": 0 # For LFU eviction policy
        }
        session_obj["H_segment"] = compute_segment_heat(session_obj)
        
        # Add to storage
        self.storage.add_mid_term_session(session_obj, processed_details)
        
        # Update local data structures
        self.sessions[session_id] = session_obj
        self.access_frequency[session_id] = 0 # Initialize for LFU
        heapq.heappush(self.heap, (-session_obj["H_segment"], session_id)) # Use negative heat for max-heap behavior
        
        print(f"MidTermMemory: Added new session {session_id}. Initial heat: {session_obj['H_segment']:.2f}.")
        if len(self.sessions) > self.max_capacity:
            self.evict_lfu()
        
        self.save()
        return session_id

    def rebuild_heap(self):
        self.heap = []
        for sid, session_data in self.sessions.items():
            # Ensure H_segment is up-to-date before rebuilding heap if necessary
            # session_data["H_segment"] = compute_segment_heat(session_data)
            heapq.heappush(self.heap, (-session_data["H_segment"], sid))
        
        # Save heap state
        self.storage.save_heap_state(self.heap)

    def insert_pages_into_session(self, summary_for_new_pages, keywords_for_new_pages, pages_to_insert, 
                                  similarity_threshold=0.6, keyword_similarity_alpha=1.0):
        if not self.sessions: # If no existing sessions, just add as a new one
            print("MidTermMemory: No existing sessions. Adding new session directly.")
            return self.add_session(summary_for_new_pages, pages_to_insert)

        new_summary_vec = self.client.get_embedding(summary_for_new_pages)
        new_summary_vec = normalize_vector(new_summary_vec)
        
        # Search for similar sessions using ChromaDB
        similar_sessions = self.storage.search_mid_term_sessions(new_summary_vec.tolist(), top_k=5)
        
        best_sid = None
        best_overall_score = -1

        for session_result in similar_sessions:
            session_id = session_result["session_id"]
            if session_id not in self.sessions:
                continue
                
            existing_session = self.sessions[session_id]
            semantic_sim = session_result["session_relevance_score"]
            
            # Keyword similarity (Jaccard index based)
            existing_keywords = set(existing_session.get("summary_keywords", []))
            new_keywords_set = set(keywords_for_new_pages)
            s_topic_keywords = 0
            if existing_keywords and new_keywords_set:
                intersection = len(existing_keywords.intersection(new_keywords_set))
                union = len(existing_keywords.union(new_keywords_set))
                if union > 0:
                    s_topic_keywords = intersection / union 
            
            overall_score = semantic_sim + keyword_similarity_alpha * s_topic_keywords
            
            if overall_score > best_overall_score:
                best_overall_score = overall_score
                best_sid = session_id
        
        if best_sid and best_overall_score >= similarity_threshold:
            print(f"MidTermMemory: Merging pages into session {best_sid}. Score: {best_overall_score:.2f} (Threshold: {similarity_threshold})")
            target_session = self.sessions[best_sid]
            
            # --- FIX: Combine new pages with existing pages instead of overwriting ---
            # 1. Get existing pages from storage backup
            existing_pages = self.storage.get_pages_from_json_backup(best_sid)
            
            # 2. Process the new pages to get embeddings, etc.
            processed_new_pages = []
            for page_data in pages_to_insert:
                page_id = page_data.get("page_id", generate_id("page"))
                
                if "page_embedding" not in page_data or not page_data["page_embedding"]:
                     full_text = f"User: {page_data.get('user_input','')} Assistant: {page_data.get('agent_response','')}"
                     page_data["page_embedding"] = normalize_vector(self.client.get_embedding(full_text)).tolist()

                if "page_keywords" not in page_data or not page_data["page_keywords"]:
                    full_text = f"User: {page_data.get('user_input','')} Assistant: {page_data.get('agent_response','')}"
                    page_data["page_keywords"] = list(extract_keywords_from_multi_summary(full_text, client=self.client,model=self.llm_model))

                processed_new_pages.append({**page_data, "page_id": page_id})

            # 3. Combine old and new pages
            all_pages_for_session = existing_pages + processed_new_pages
            
            # 4. Update session metadata
            target_session["L_interaction"] += len(processed_new_pages)
            target_session["last_visit_time"] = get_timestamp()
            target_session["N_visit"] += 1
            target_session["H_segment"] = compute_segment_heat(target_session)
            
            # 5. Add the updated session and the complete list of pages to storage
            self.storage.add_mid_term_session(target_session, all_pages_for_session)
            
            # Update local heap
            self.rebuild_heap()
            print(f"MidTermMemory: Merged {len(processed_new_pages)} new pages into session {best_sid}. Total pages now: {len(all_pages_for_session)}.")

        else:
            # If no suitable session found, add as a new session
            print("MidTermMemory: No suitable session found. Adding as a new session.")
            self.add_session(summary_for_new_pages, pages_to_insert)

    def search_sessions(self, query_text, segment_similarity_threshold=0.1, page_similarity_threshold=0.1, 
                          top_k_sessions=5, keyword_alpha=1.0, recency_tau_search=3600):
        if not self.sessions:
            return []

        query_vec = self.client.get_embedding(query_text)
        query_vec = normalize_vector(query_vec)
        query_keywords = set(extract_keywords_from_multi_summary(query_text, client=self.client,model=self.llm_model))

        # Search sessions using ChromaDB
        similar_sessions = self.storage.search_mid_term_sessions(query_vec.tolist(), top_k=top_k_sessions)

        results = []
        current_time_str = get_timestamp()

        for session_result in similar_sessions:
            session_id = session_result["session_id"]
            if session_id not in self.sessions:
                continue
                
            session = self.sessions[session_id]
            semantic_sim_score = session_result["session_relevance_score"]

            # Keyword similarity for session summary
            session_keywords = set(session.get("summary_keywords", []))
            s_topic_keywords = 0
            if query_keywords and session_keywords:
                intersection = len(query_keywords.intersection(session_keywords))
                union = len(query_keywords.union(session_keywords))
                if union > 0: s_topic_keywords = intersection / union
            
            # Combined score for session relevance
            session_relevance_score = semantic_sim_score + keyword_alpha * s_topic_keywords

            if session_relevance_score >= segment_similarity_threshold:
                # Search pages within this session
                matched_pages = self.storage.search_mid_term_pages(query_vec.tolist(), [session_id], top_k=20)
                
                # Filter pages by similarity threshold
                filtered_pages = []
                for page_result in matched_pages:
                    if page_result["relevance_score"] >= page_similarity_threshold:
                        filtered_pages.append(page_result)
                
                if filtered_pages:
                    # Update session access stats
                    session["N_visit"] += 1
                    session["last_visit_time"] = current_time_str
                    session["access_count_lfu"] = session.get("access_count_lfu", 0) + 1
                    self.access_frequency[session_id] = session["access_count_lfu"]
                    session["H_segment"] = compute_segment_heat(session)
                    
                    # Update storage
                    self.storage.update_mid_term_session_metadata(session_id, {
                        "N_visit": session["N_visit"],
                        "last_visit_time": session["last_visit_time"],
                        "access_count_lfu": session["access_count_lfu"],
                        "H_segment": session["H_segment"]
                    })
                    
                    self.rebuild_heap() # Heat changed
                    
                    results.append({
                        "session_id": session_id,
                        "session_summary": session["summary"],
                        "session_relevance_score": session_relevance_score,
                        "matched_pages": filtered_pages
                    })
        
        self.save() # Save changes from access updates
        # Sort final results by session_relevance_score
        return sorted(results, key=lambda x: x["session_relevance_score"], reverse=True)

    def save(self):
        # Save access frequency and heap state
        for session_id, freq in self.access_frequency.items():
            self.storage.update_access_frequency(session_id, freq)
        
        self.storage.save_heap_state(self.heap) 