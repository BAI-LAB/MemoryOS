import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# 修改为绝对导入
try:
    # 尝试相对导入（当作为包使用时）
    from .utils import OpenAIClient, get_timestamp, generate_id, gpt_user_profile_analysis, gpt_knowledge_extraction, ensure_directory_exists
    from . import prompts
    from .short_term import ShortTermMemory
    from .mid_term import MidTermMemory, compute_segment_heat # For H_THRESHOLD logic
    from .long_term import LongTermMemory
    from .updater import Updater
    from .retriever import Retriever
except ImportError:
    # 回退到绝对导入（当作为独立模块使用时）
    from utils import OpenAIClient, get_timestamp, generate_id, gpt_user_profile_analysis, gpt_knowledge_extraction, ensure_directory_exists
    import prompts
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
                 openai_base_url: str = None, 
                 assistant_id: str = DEFAULT_ASSISTANT_ID, 
                 short_term_capacity=10,
                 mid_term_capacity=2000,
                 long_term_knowledge_capacity=100,
                 retrieval_queue_capacity=7,
                 mid_term_heat_threshold=H_PROFILE_UPDATE_THRESHOLD,
                 mid_term_similarity_threshold=0.6,
                 llm_model="gpt-4o-mini",
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 embedding_model_kwargs: dict = None
                 ):
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.data_storage_path = os.path.abspath(data_storage_path)
        self.llm_model = llm_model
        self.mid_term_similarity_threshold = mid_term_similarity_threshold
        self.embedding_model_name = embedding_model_name
        
        # Smart defaults for embedding_model_kwargs
        if embedding_model_kwargs is None:
            if 'bge-m3' in self.embedding_model_name.lower():
                print("INFO: Detected bge-m3 model, defaulting embedding_model_kwargs to {'use_fp16': True}")
                self.embedding_model_kwargs = {'use_fp16': True}
            else:
                self.embedding_model_kwargs = {}
        else:
            self.embedding_model_kwargs = embedding_model_kwargs


        print(f"Initializing Memoryos for user '{self.user_id}' and assistant '{self.assistant_id}'. Data path: {self.data_storage_path}")
        print(f"Using unified LLM model: {self.llm_model}")
        print(f"Using embedding model: {self.embedding_model_name} with kwargs: {self.embedding_model_kwargs}")

        # Initialize OpenAI Client
        self.client = OpenAIClient(api_key=openai_api_key, base_url=openai_base_url)

        # Define file paths for user-specific data
        self.user_data_dir = os.path.join(self.data_storage_path, "users", self.user_id)
        user_short_term_path = os.path.join(self.user_data_dir, "short_term.json")
        user_mid_term_path = os.path.join(self.user_data_dir, "mid_term.json")
        user_long_term_path = os.path.join(self.user_data_dir, "long_term_user.json") # User profile and their knowledge

        # Define file paths for assistant-specific data (knowledge)
        self.assistant_data_dir = os.path.join(self.data_storage_path, "assistants", self.assistant_id)
        assistant_long_term_path = os.path.join(self.assistant_data_dir, "long_term_assistant.json")

        # Ensure directories exist
        ensure_directory_exists(user_short_term_path) # ensure_directory_exists operates on the file path, creating parent dirs
        ensure_directory_exists(user_mid_term_path)
        ensure_directory_exists(user_long_term_path)
        ensure_directory_exists(assistant_long_term_path)

        # Initialize Memory Modules for User
        self.short_term_memory = ShortTermMemory(file_path=user_short_term_path, max_capacity=short_term_capacity)
        self.mid_term_memory = MidTermMemory(
            file_path=user_mid_term_path, 
            client=self.client, 
            max_capacity=mid_term_capacity,
            embedding_model_name=self.embedding_model_name,
            embedding_model_kwargs=self.embedding_model_kwargs
        )
        self.user_long_term_memory = LongTermMemory(
            file_path=user_long_term_path, 
            knowledge_capacity=long_term_knowledge_capacity,
            embedding_model_name=self.embedding_model_name,
            embedding_model_kwargs=self.embedding_model_kwargs
        )

        # Initialize Memory Module for Assistant Knowledge
        self.assistant_long_term_memory = LongTermMemory(
            file_path=assistant_long_term_path, 
            knowledge_capacity=long_term_knowledge_capacity,
            embedding_model_name=self.embedding_model_name,
            embedding_model_kwargs=self.embedding_model_kwargs
        )

        # Initialize Orchestration Modules
        self.updater = Updater(short_term_memory=self.short_term_memory, 
                               mid_term_memory=self.mid_term_memory, 
                               long_term_memory=self.user_long_term_memory, # Updater primarily updates user's LTM profile/knowledge
                               client=self.client,
                               topic_similarity_threshold=mid_term_similarity_threshold,  # 传递中期记忆相似度阈值
                               llm_model=self.llm_model)
        self.retriever = Retriever(
            mid_term_memory=self.mid_term_memory,
            long_term_memory=self.user_long_term_memory,
            assistant_long_term_memory=self.assistant_long_term_memory, # Pass assistant LTM
            queue_capacity=retrieval_queue_capacity
        )
        
        self.mid_term_heat_threshold = mid_term_heat_threshold

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
            # A page is a dict: {"user_input": ..., "agent_response": ..., "timestamp": ..., "analyzed": False, ...}
            unanalyzed_pages = [p for p in session.get("details", []) if not p.get("analyzed", False)]

            if unanalyzed_pages:
                print(f"Memoryos: Mid-term session {sid} heat ({current_heat:.2f}) exceeded threshold. Analyzing {len(unanalyzed_pages)} pages for profile/knowledge update.")
                
                # 并行执行两个LLM任务：用户画像分析（已包含更新）、知识提取
                def task_user_profile_analysis():
                    print("Memoryos: Starting parallel user profile analysis and update...")
                    # 获取现有用户画像
                    existing_profile = self.user_long_term_memory.get_raw_user_profile(self.user_id)
                    if not existing_profile or existing_profile.lower() == "none":
                        existing_profile = "No existing profile data."
                    
                    # 直接输出更新后的完整画像
                    return gpt_user_profile_analysis(unanalyzed_pages, self.client, model=self.llm_model, existing_user_profile=existing_profile)
                
                def task_knowledge_extraction():
                    print("Memoryos: Starting parallel knowledge extraction...")
                    return gpt_knowledge_extraction(unanalyzed_pages, self.client, model=self.llm_model)
                
                # 使用并行任务执行                
                with ThreadPoolExecutor(max_workers=2) as executor:
                    # 提交两个主要任务
                    future_profile = executor.submit(task_user_profile_analysis)
                    future_knowledge = executor.submit(task_knowledge_extraction)
                    
                    # 等待结果
                    try:
                        updated_user_profile = future_profile.result()  # 直接是更新后的完整画像
                        knowledge_result = future_knowledge.result()
                    except Exception as e:
                        print(f"Error in parallel LLM processing: {e}")
                        return
                
                new_user_private_knowledge = knowledge_result.get("private")
                new_assistant_knowledge = knowledge_result.get("assistant_knowledge")

                # 直接使用更新后的完整用户画像
                if updated_user_profile and updated_user_profile.lower() != "none":
                    print("Memoryos: Updating user profile with integrated analysis...")
                    self.user_long_term_memory.update_user_profile(self.user_id, updated_user_profile, merge=False)  # 直接替换为新的完整画像
                
                # Add User Private Knowledge to user's LTM
                if new_user_private_knowledge and new_user_private_knowledge.lower() != "none":
                    for line in new_user_private_knowledge.split('\n'):
                         if line.strip() and line.strip().lower() not in ["none", "- none", "- none."]:
                            self.user_long_term_memory.add_user_knowledge(line.strip())

                # Add Assistant Knowledge to assistant's LTM
                if new_assistant_knowledge and new_assistant_knowledge.lower() != "none":
                    for line in new_assistant_knowledge.split('\n'):
                        if line.strip() and line.strip().lower() not in ["none", "- none", "- none."]:
                           self.assistant_long_term_memory.add_assistant_knowledge(line.strip()) # Save to dedicated assistant LTM

                # Mark pages as analyzed and reset session heat contributors
                for p in session["details"]:
                    p["analyzed"] = True # Mark all pages in session, or just unanalyzed_pages?
                                          # Original code marked all pages in session
                
                session["N_visit"] = 0 # Reset visits after analysis
                session["L_interaction"] = 0 # Reset interaction length contribution
                # session["R_recency"] = 1.0 # Recency will re-calculate naturally
                session["H_segment"] = compute_segment_heat(session) # Recompute heat with reset factors
                session["last_visit_time"] = get_timestamp() # Update last visit time
                
                self.mid_term_memory.rebuild_heap() # Heap needs rebuild due to H_segment change
                self.mid_term_memory.save()
                print(f"Memoryos: Profile/Knowledge update for session {sid} complete. Heat reset.")
            else:
                print(f"Memoryos: Hot session {sid} has no unanalyzed pages. Skipping profile update.")
        else:
            # print(f"Memoryos: Top session {sid} heat ({current_heat:.2f}) below threshold. No profile update.")
            pass # No action if below threshold

    def add_memory(self, user_input: str, agent_response: str, timestamp: str = None, meta_data: dict = None):
        """
        Adds a new QA pair (memory) to the system.
        meta_data is not used in the current refactoring but kept for future use.
        """
        if not timestamp:
            timestamp = get_timestamp()
        
        qa_pair = {
            "user_input": user_input,
            "agent_response": agent_response,
            "timestamp": timestamp
            # meta_data can be added here if it needs to be stored with the QA pair
        }
        self.short_term_memory.add_qa_pair(qa_pair)
        print(f"Memoryos: Added QA to short-term. User: {user_input[:30]}...")

        if self.short_term_memory.is_full():
            print("Memoryos: Short-term memory full. Processing to mid-term.")
            self.updater.process_short_term_to_mid_term()
        
        # After any memory addition that might impact mid-term, check for profile updates
        self._trigger_profile_and_knowledge_update_if_needed()

    def get_response(self, query: str, relationship_with_user="friend", style_hint="", user_conversation_meta_data: dict = None) -> str:
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

        # 3. Format retrieved mid-term pages (retrieval_queue equivalent)
        retrieval_text = "\n".join([
            f"【Historical Memory】\nUser: {page.get('user_input', '')}\nAssistant: {page.get('agent_response', '')}\nTime: {page.get('timestamp', '')}\nConversation chain overview: {page.get('meta_info','N/A')}"
            for page in retrieved_pages
        ])

        # 4. Get user profile
        user_profile_text = self.user_long_term_memory.get_raw_user_profile(self.user_id)
        if not user_profile_text or user_profile_text.lower() == "none": 
            user_profile_text = "No detailed profile available yet."

        # 5. Format retrieved user knowledge for background
        user_knowledge_background = ""
        if retrieved_user_knowledge:
            user_knowledge_background = "\n【Relevant User Knowledge Entries】\n"
            for kn_entry in retrieved_user_knowledge:
                user_knowledge_background += f"- {kn_entry['knowledge']} (Recorded: {kn_entry['timestamp']})\n"
        
        background_context = f"【User Profile】\n{user_profile_text}\n{user_knowledge_background}"

        # 6. Format retrieved Assistant Knowledge (from assistant's LTM)
        # Use retrieved assistant knowledge instead of all assistant knowledge
        assistant_knowledge_text_for_prompt = "【Assistant Knowledge Base】\n"
        if retrieved_assistant_knowledge:
            for ak_entry in retrieved_assistant_knowledge:
                assistant_knowledge_text_for_prompt += f"- {ak_entry['knowledge']} (Recorded: {ak_entry['timestamp']})\n"
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
    def get_user_profile_summary(self) -> str:
        return self.user_long_term_memory.get_raw_user_profile(self.user_id)

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