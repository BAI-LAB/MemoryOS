import os
import json
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed

# 修改为绝对导入
try:
    # 尝试相对导入（当作为包使用时）
    from .utils import OpenAIClient, get_timestamp, generate_id, gpt_user_profile_analysis, gpt_knowledge_extraction, ensure_directory_exists
    from . import prompts
    from .storage_provider import ChromaStorageProvider
    from .short_term import ShortTermMemory
    from .mid_term import MidTermMemory, compute_segment_heat # For H_THRESHOLD logic
    from .long_term import LongTermMemory
    from .updater import Updater
    from .retriever import Retriever
except ImportError:
    # 回退到绝对导入（当作为独立模块使用时）
    from utils import OpenAIClient, get_timestamp, generate_id, gpt_user_profile_analysis, gpt_knowledge_extraction, ensure_directory_exists
    import prompts
    from storage_provider import ChromaStorageProvider
    from short_term import ShortTermMemory
    from mid_term import MidTermMemory, compute_segment_heat # For H_THRESHOLD logic
    from long_term import LongTermMemory
    from updater import Updater
    from retriever import Retriever

# Heat threshold for triggering profile/knowledge update from mid-term memory
H_PROFILE_UPDATE_THRESHOLD = 5.0 
DEFAULT_ASSISTANT_ID = "default_assistant_profile"

class Memoryos:
    def __init__(self, user_id: str, 
                 openai_api_key: str, 
                 data_storage_path: str,
                 openai_base_url = None, 
                 assistant_id: str = DEFAULT_ASSISTANT_ID, 
                 short_term_capacity=10,
                 mid_term_capacity=2000,
                 long_term_knowledge_capacity=100,
                 retrieval_queue_capacity=7,
                 mid_term_heat_threshold=H_PROFILE_UPDATE_THRESHOLD,
                 mid_term_similarity_threshold=0.6,
                 llm_model="gpt-4o-mini",
                 embedding_model_name: str = "text-embedding-3-small",
                 embedding_api_key: str = None,
                 embedding_base_url: str = None
                 ):
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.data_storage_path = os.path.abspath(data_storage_path)
        self.llm_model = llm_model
        self.mid_term_similarity_threshold = mid_term_similarity_threshold
        self.embedding_model_name = embedding_model_name

        print(f"Initializing Memoryos for user '{self.user_id}' and assistant '{self.assistant_id}'. Data path: {self.data_storage_path}")
        print(f"Using unified LLM model: {self.llm_model}")
        print(f"Using embedding model: {self.embedding_model_name}")

        # Initialize OpenAI Client
        self.client = OpenAIClient(api_key=openai_api_key, base_url=openai_base_url, embedding_api_key=embedding_api_key, embedding_base_url=embedding_base_url, embedding_model=embedding_model_name)
        
        # Centralized Storage Provider
        storage_path = os.path.join(self.data_storage_path, "chroma_storage")
        self.storage_provider = ChromaStorageProvider(
            path=storage_path, 
            user_id=self.user_id, 
            assistant_id=self.assistant_id
        )

        # Register save handler to be called on exit
        atexit.register(self.close)

        # Initialize Memory Modules with the shared storage provider
        self.short_term_memory = ShortTermMemory(
            storage_provider=self.storage_provider,
            max_capacity=short_term_capacity
        )
        self.mid_term_memory = MidTermMemory(
            storage_provider=self.storage_provider,
            user_id=self.user_id,
            client=self.client,
            max_capacity=mid_term_capacity,
            llm_model=self.llm_model
        )
        self.user_long_term_memory = LongTermMemory(
            storage_provider=self.storage_provider,
            llm_interface=self.client,
            client=self.client,
            llm_model=self.llm_model
        )

        # Initialize Memory Module for Assistant Knowledge
        self.assistant_long_term_memory = LongTermMemory(
            storage_provider=self.storage_provider,
            llm_interface=self.client,
            client=self.client,
            llm_model=self.llm_model
        )

        # Initialize Orchestration Modules
        self.updater = Updater(
            short_term_memory=self.short_term_memory, 
            mid_term_memory=self.mid_term_memory, 
            long_term_memory=self.user_long_term_memory,
            client=self.client,
            topic_similarity_threshold=mid_term_similarity_threshold,
            llm_model=self.llm_model
        )
        self.retriever = Retriever(
            mid_term_memory=self.mid_term_memory,
            user_long_term_memory=self.user_long_term_memory,
            assistant_long_term_memory=self.assistant_long_term_memory,
            queue_capacity=retrieval_queue_capacity
        )
        
        self.mid_term_heat_threshold = mid_term_heat_threshold

    def close(self):
        """Saves all metadata to disk. Registered with atexit to be called on script termination."""
        print("Memoryos: Process is terminating. Saving all metadata to disk...")
        self.storage_provider.save_all_metadata()
        print("Memoryos: Metadata saved successfully.")

    def _trigger_profile_and_knowledge_update_if_needed(self):
        """
        Checks mid-term memory for hot segments and triggers profile/knowledge update if threshold is met.
        Adapted from main_memoybank.py's update_user_profile_from_top_segment.
        Enhanced with parallel LLM processing for better performance.
        """
        if not self.mid_term_memory.heap:
            return

        # Peek at the top of the heap (hottest segment)
        # MidTermMemory heap stores (-H_segment, sid)
        neg_heat, sid = self.mid_term_memory.heap[0] 
        current_heat = -neg_heat

        if current_heat >= self.mid_term_heat_threshold:
            session = self.mid_term_memory.sessions.get(sid)
            if not session:
                self.mid_term_memory.rebuild_heap() # Clean up if session is gone
                return

            # Get unanalyzed pages from this hot session
            unanalyzed_pages = [
                page for page in self.mid_term_memory.storage.get_pages_from_json_backup(sid)
                if not page.get("analyzed")
            ]

            if unanalyzed_pages:
                print(f"Memoryos: Mid-term session {sid} heat ({current_heat:.2f}) exceeded threshold. Analyzing {len(unanalyzed_pages)} pages for profile/knowledge update.")
                
                # Combine all unanalyzed page interactions into a single string for LLM
                conversation_str = "\n".join(
                    [f"User: {p.get('user_input', '')}\nAssistant: {p.get('agent_response', '')}" for p in unanalyzed_pages]
                )

                def task_update_profile():
                    print("Memoryos: Starting user profile update task...")
                    return self.user_long_term_memory.update_user_profile(self.user_id, conversation_str)

                def task_extract_knowledge():
                    print("Memoryos: Starting knowledge extraction task...")
                    # This function needs the raw conversation string from the hot pages
                    return self.user_long_term_memory.extract_knowledge_from_text(conversation_str)

                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_profile = executor.submit(task_update_profile)
                    future_knowledge = executor.submit(task_extract_knowledge)

                    try:
                        updated_profile = future_profile.result()
                        knowledge_result = future_knowledge.result()
                    except Exception as e:
                        print(f"Error in parallel LLM processing: {e}")
                        return

                # The profile is already updated in memory by update_user_profile
                if updated_profile:
                    self.storage_provider.record_update_time("profile", get_timestamp())
                    print(f"Memoryos: User profile update recorded in memory for user {self.user_id}.")
                
                # Add extracted knowledge
                if knowledge_result:
                    user_knowledge = knowledge_result.get("private")
                    if user_knowledge:
                        # Ensure user_knowledge is a list before iterating
                        if isinstance(user_knowledge, str):
                            user_knowledge = [user_knowledge]
                        for item in user_knowledge:
                            self.user_long_term_memory.add_knowledge(item, "user")
                    
                    assistant_knowledge = knowledge_result.get("assistant_knowledge")
                    if assistant_knowledge:
                        # Ensure assistant_knowledge is a list before iterating
                        if isinstance(assistant_knowledge, str):
                            assistant_knowledge = [assistant_knowledge]
                        for item in assistant_knowledge:
                            self.assistant_long_term_memory.add_knowledge(item, "assistant")

                    self.storage_provider.record_update_time("knowledge", get_timestamp())

                # Mark pages as analyzed and reset session heat
                for page in unanalyzed_pages:
                    page["analyzed"] = True
                
                # Update the session metadata in storage
                session["N_visit"] = 0 
                session["L_interaction"] = 0
                session["H_segment"] = compute_segment_heat(session)
                session["last_visit_time"] = get_timestamp()
                self.mid_term_memory.storage.update_mid_term_session_metadata(sid, session)

                self.mid_term_memory.rebuild_heap()
                print(f"Memoryos: Profile/Knowledge update for session {sid} complete. Heat reset.")
            else:
                print(f"Memoryos: Hot session {sid} has no unanalyzed pages. Skipping profile update.")
        else:
            # print(f"Memoryos: Top session {sid} heat ({current_heat:.2f}) below threshold. No profile update.")
            pass # No action if below threshold

    def add_memory(self, user_input: str, agent_response: str, timestamp = None, meta_data = None):
        """
        Adds a new QA pair (memory) to the system.
        meta_data is not used in the current refactoring but kept for future use.
        """
        if not timestamp:
            timestamp = get_timestamp()
        
        qa_pair = {
            "user_id": self.user_id, # Add user_id to qa_pair
            "user_input": user_input,
            "agent_response": agent_response,
            "timestamp": timestamp
        }
        self.short_term_memory.add_qa_pair(qa_pair)
        print(f"Memoryos: Added QA to short-term. User: {user_input[:30]}...")

        if self.short_term_memory.is_full():
            print("Memoryos: Short-term memory full. Processing to mid-term.")
            self.updater.process_short_term_to_mid_term()
        
        # After any memory addition that might impact mid-term, check for profile updates
        self._trigger_profile_and_knowledge_update_if_needed()

    def get_response(self, query: str, relationship_with_user="friend", style_hint="", user_conversation_meta_data = None) -> str:
        """
        Generates a response to the user's query, incorporating memory and context.
        """
        print(f"Memoryos: Generating response for query: '{query[:50]}...'")

        # 1. Retrieve context
        retrieval_results = self.retriever.retrieve_context(
            user_query=query,
            user_id=self.user_id
            # Using default thresholds from Retriever class for now
        )
        retrieved_pages = retrieval_results["retrieved_pages"]
        retrieved_user_knowledge = retrieval_results["retrieved_user_knowledge"]
        retrieved_assistant_knowledge = retrieval_results["retrieved_assistant_knowledge"]

        # 2. Get short-term history
        short_term_history = self.short_term_memory.get_all()
        history_text = "\n".join([
            f"User: {qa.get('user_input', '')}\nAssistant: {qa.get('agent_response', '')} (Time: {qa.get('timestamp', '')})"
            for qa in short_term_history
        ])

        # 3. Format retrieved mid-term pages with dialogue chain info
        retrieval_text_parts = []
        for p in retrieved_pages:
            # 获取页面的完整信息，包括meta_info
            page_id = p.get('page_id', '')
            session_id = p.get('session_id', '')
            
            # 从JSON备份中获取meta_info
            meta_info = ""
            if page_id and session_id:
                full_page_info = self.storage_provider.get_page_full_info(page_id, session_id)
                if full_page_info:
                    meta_info = full_page_info.get('meta_info', '')
            
            # 构建包含对话链信息的文本
            page_text = f"User: {p.get('user_input', '')}\nAssistant: {p.get('agent_response', '')}"
            if meta_info:
                page_text += f"\n Dialogue chain info: \n{meta_info}"
            
            retrieval_text_parts.append(page_text)
        retrieval_text="【Historical Memory】\n"
        retrieval_text += "\n\n".join(retrieval_text_parts)

        # 4. Get user profile
        user_profile_data = self.user_long_term_memory.get_user_profile(self.user_id)
        user_profile_text = json.dumps(user_profile_data, indent=2, ensure_ascii=False) if user_profile_data else "No detailed profile available yet."

        # 5. Format retrieved user knowledge
        user_knowledge_background = ""
        if retrieved_user_knowledge:
            user_knowledge_background = "\n【Relevant User Knowledge】\n"
            for kn_entry in retrieved_user_knowledge:
                user_knowledge_background += f"- {kn_entry['text']}\n"
        
        background_context = f"【User Profile】\n{user_profile_text}\n{user_knowledge_background}"

        # 6. Format retrieved Assistant Knowledge
        assistant_knowledge_text_for_prompt = "【Assistant Knowledge Base】\n"
        if retrieved_assistant_knowledge:
            for ak_entry in retrieved_assistant_knowledge:
                assistant_knowledge_text_for_prompt += f"- {ak_entry['text']}\n"
        else:
            assistant_knowledge_text_for_prompt += "- No relevant assistant knowledge found for this query.\n"

        # 7. Format user_conversation_meta_data (if provided)
        meta_data_text_for_prompt = "【Current Conversation Metadata】\n"
        if user_conversation_meta_data:
            try:
                meta_data_text_for_prompt += json.dumps(user_conversation_meta_data, ensure_ascii=False, indent=2)
            except TypeError:
                meta_data_text_for_prompt += str(user_conversation_meta_data)
        else:
            meta_data_text_for_prompt += "None provided for this turn."

        # 8. Construct Prompts
        system_prompt_text = prompts.GENERATE_SYSTEM_RESPONSE_SYSTEM_PROMPT.format(
            relationship=relationship_with_user,
            assistant_knowledge_text=assistant_knowledge_text_for_prompt,
            meta_data_text=meta_data_text_for_prompt # Using meta_data_text placeholder for user_conversation_meta_data
        )
        
        user_prompt_text = prompts.GENERATE_SYSTEM_RESPONSE_USER_PROMPT.format(
            history_text=history_text,
            retrieval_text=retrieval_text,
            background=background_context,
            relationship=relationship_with_user,
            query=query
        )
        
        messages = [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": user_prompt_text}
        ]

        # 9. Call LLM for response
        print("Memoryos: Calling LLM for final response generation...")
        # print("System Prompt:\n", system_prompt_text)
        # print("User Prompt:\n", user_prompt_text)
        response_content = self.client.chat_completion(
            model=self.llm_model, 
            messages=messages, 
            temperature=0.7, 
            max_tokens=1500 # As in original main
        )
        
        # 10. Add this interaction to memory
        self.add_memory(user_input=query, agent_response=response_content, timestamp=get_timestamp())
        
        return response_content

    # --- Helper/Maintenance methods (optional additions) ---
    def get_user_profile_summary(self) -> dict:
        """Retrieves the full user profile object."""
        profile = self.user_long_term_memory.get_user_profile(self.user_id)
        return profile or {}

    def get_assistant_knowledge_summary(self) -> list:
        return self.assistant_long_term_memory.get_assistant_knowledge()

    def force_mid_term_analysis(self):
        """Forces analysis of all unanalyzed pages in the hottest mid-term segment if heat is above 0.
           Useful for testing or manual triggering.
        """
        original_threshold = self.mid_term_heat_threshold
        self.mid_term_heat_threshold = 0.0 # Temporarily lower threshold
        print("Memoryos: Force-triggering mid-term analysis...")
        self._trigger_profile_and_knowledge_update_if_needed()
        self.mid_term_heat_threshold = original_threshold # Restore original threshold

    def __repr__(self):
        return f"<Memoryos user_id='{self.user_id}' assistant_id='{self.assistant_id}' data_path='{self.data_storage_path}'>" 