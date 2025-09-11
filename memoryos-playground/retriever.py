from collections import deque
import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

try:
    from .utils import get_timestamp, OpenAIClient, run_parallel_tasks
    from .short_term import ShortTermMemory
    from .mid_term import MidTermMemory
    from .long_term import LongTermMemory
except ImportError:
    from utils import get_timestamp, OpenAIClient, run_parallel_tasks
    from short_term import ShortTermMemory
    from mid_term import MidTermMemory
    from long_term import LongTermMemory
# from .updater import Updater # Updater is not directly used by Retriever

class Retriever:
    def __init__(self, 
                 mid_term_memory: MidTermMemory, 
                 long_term_memory: LongTermMemory, 
                 assistant_long_term_memory: Optional[LongTermMemory] = None, # Add assistant LTM
                 # client: OpenAIClient, # Not strictly needed if all LLM calls are within memory modules
                 queue_capacity=7): # Default from main_memoybank was 7 for retrieval_queue
        # Short term memory is usually for direct context, not primary retrieval source here
        # self.short_term_memory = short_term_memory 
        self.mid_term_memory = mid_term_memory
        self.long_term_memory = long_term_memory
        self.assistant_long_term_memory = assistant_long_term_memory # Store assistant LTM reference
        # self.client = client 
        self.retrieval_queue_capacity = queue_capacity
        # self.retrieval_queue = deque(maxlen=queue_capacity) # This was instance level, but retrieve returns it, so maybe not needed as instance var

    def _retrieve_mid_term_context(self, user_query, segment_similarity_threshold, page_similarity_threshold, top_k_sessions):
        """并行任务：从中期记忆检索"""
        print("Retriever: Searching mid-term memory...")
        matched_sessions = self.mid_term_memory.search_sessions(
            query_text=user_query, 
            segment_similarity_threshold=segment_similarity_threshold,
            page_similarity_threshold=page_similarity_threshold,
            top_k_sessions=top_k_sessions
        )
        
        # Use a heap to get top N pages across all relevant sessions based on their scores
        top_pages_heap = []
        page_counter = 0  # Add counter to ensure unique comparison
        for session_match in matched_sessions:
            for page_match in session_match.get("matched_pages", []):
                page_data = page_match["page_data"]
                page_score = page_match["score"] # Using the page score directly
                
                # Add session relevance score to page score or combine them?
                # For now, using page_score. Could be: page_score * session_match["session_relevance_score"]
                combined_score = page_score # Potentially adjust with session_relevance_score

                if len(top_pages_heap) < self.retrieval_queue_capacity:
                    heapq.heappush(top_pages_heap, (combined_score, page_counter, page_data))
                    page_counter += 1
                elif combined_score > top_pages_heap[0][0]: # If current page is better than the worst in heap
                    heapq.heappop(top_pages_heap)
                    heapq.heappush(top_pages_heap, (combined_score, page_counter, page_data))
                    page_counter += 1
        
        # Extract pages from heap, already sorted by heapq property (smallest first)
        # We want highest scores, so either use a max-heap or sort after popping from min-heap.
        retrieved_pages = [item[2] for item in sorted(top_pages_heap, key=lambda x: x[0], reverse=True)]
        print(f"Retriever: Mid-term memory recalled {len(retrieved_pages)} pages.")
        return retrieved_pages

    def _retrieve_user_knowledge(self, user_query, knowledge_threshold, top_k_knowledge):
        """并行任务：从用户长期知识检索"""
        print("Retriever: Searching user long-term knowledge...")
        retrieved_knowledge = self.long_term_memory.search_user_knowledge(
            user_query, threshold=knowledge_threshold, top_k=top_k_knowledge
        )
        print(f"Retriever: Long-term user knowledge recalled {len(retrieved_knowledge)} items.")
        return retrieved_knowledge

    def _retrieve_assistant_knowledge(self, user_query, knowledge_threshold, top_k_knowledge):
        """并行任务：从助手长期知识检索"""
        if not self.assistant_long_term_memory:
            print("Retriever: No assistant long-term memory provided, skipping assistant knowledge retrieval.")
            return []
        
        print("Retriever: Searching assistant long-term knowledge...")
        retrieved_knowledge = self.assistant_long_term_memory.search_assistant_knowledge(
            user_query, threshold=knowledge_threshold, top_k=top_k_knowledge
        )
        print(f"Retriever: Long-term assistant knowledge recalled {len(retrieved_knowledge)} items.")
        return retrieved_knowledge

    def retrieve_context(self, user_query: str, 
                         user_id: str, # Needed for profile, can be used for context filtering if desired
                         segment_similarity_threshold=0.1,  # From main_memoybank example
                         page_similarity_threshold=0.1,     # From main_memoybank example
                         knowledge_threshold=0.01,          # From main_memoybank example
                         top_k_sessions=5,                  # From MidTermMemory search default
                         top_k_knowledge=20                  # Default for knowledge search
                         ):
        print(f"Retriever: Starting PARALLEL retrieval for query: '{user_query[:50]}...'")
        
        # 并行执行三个检索任务
        tasks = [
            lambda: self._retrieve_mid_term_context(user_query, segment_similarity_threshold, page_similarity_threshold, top_k_sessions),
            lambda: self._retrieve_user_knowledge(user_query, knowledge_threshold, top_k_knowledge),
            lambda: self._retrieve_assistant_knowledge(user_query, knowledge_threshold, top_k_knowledge)
        ]
        
        # 使用并行处理
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i, task in enumerate(tasks):
                future = executor.submit(task)
                futures.append((i, future))
            
            results = [None] * 3
            for task_idx, future in futures:
                try:
                    results[task_idx] = future.result()
                except Exception as e:
                    print(f"Error in retrieval task {task_idx}: {e}")
                    results[task_idx] = []
        
        retrieved_mid_term_pages, retrieved_user_knowledge, retrieved_assistant_knowledge = results

        return {
            "retrieved_pages": retrieved_mid_term_pages or [], # List of page dicts
            "retrieved_user_knowledge": retrieved_user_knowledge or [], # List of knowledge entry dicts
            "retrieved_assistant_knowledge": retrieved_assistant_knowledge or [], # List of assistant knowledge entry dicts
            "retrieved_at": get_timestamp()
        } 