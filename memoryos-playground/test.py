
import os
from memoryos import Memoryos

# --- Basic Configuration ---
USER_ID = "demo_user"
ASSISTANT_ID = "demo_assistant"
API_KEY = "sk-7VaFJuGM146a957c4E75T3BlBkFJb7232107783F41C29e00"  # Replace with your key
BASE_URL = "https://cn2us02.opapi.win/v1"  # Optional: if using a custom OpenAI endpoint
DATA_STORAGE_PATH = "./simple_demo_data"
LLM_MODEL = "gpt-4o-mini"

def simple_demo():
    print("MemoryOS Simple Demo")
    
    # 1. Initialize MemoryOS
    print("Initializing MemoryOS...")
    try:
        memo = Memoryos(
            user_id=USER_ID,
            openai_api_key=API_KEY,
            openai_base_url=BASE_URL,
            data_storage_path=DATA_STORAGE_PATH,
            llm_model=LLM_MODEL,
            assistant_id=ASSISTANT_ID,
            short_term_capacity=7,  
            mid_term_heat_threshold=1000,  
            retrieval_queue_capacity=10,
            long_term_knowledge_capacity=100,
            mid_term_similarity_threshold=0.6,
            embedding_model_name="/root/autodl-tmp/embedding_cache/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181/"
        )
        print("MemoryOS initialized successfully!\n")
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Add some basic memories
    print("Adding some memories...")
    
    memo.add_memory(
        user_input="Hi! I'm Tom, I work as a data scientist in San Francisco.",
        agent_response="Hello Tom! Nice to meet you. Data science is such an exciting field. What kind of data do you work with?"
    )
    memo.add_memory(
        user_input="I love hiking on weekends, especially in the mountains.",
        agent_response="That sounds wonderful! Do you have a favorite trail or mountain you like to visit?"
    )
    memo.add_memory(
        user_input="Recently, I've been reading a lot about artificial intelligence.",
        agent_response="AI is a fascinating topic! Are you interested in any specific area of AI?"
    )
    memo.add_memory(
        user_input="My favorite food is sushi, especially salmon nigiri.",
        agent_response="Sushi is delicious! Have you ever tried making it at home?"
    )
    memo.add_memory(
        user_input="I have a golden retriever named Max.",
        agent_response="Max must be adorable! How old is he?"
    )
    memo.add_memory(
        user_input="I traveled to Japan last year and visited Tokyo and Kyoto.",
        agent_response="That must have been an amazing experience! What did you enjoy most about Japan?"
    )
    memo.add_memory(
        user_input="I'm currently learning how to play the guitar.",
        agent_response="That's awesome! What songs are you practicing right now?"
    )
    memo.add_memory(
        user_input="I usually start my day with a cup of black coffee.",
        agent_response="Coffee is a great way to kickstart the day! Do you prefer it hot or iced?"
    )
    memo.add_memory(
        user_input="My favorite movie genre is science fiction.",
        agent_response="Sci-fi movies can be so imaginative! Do you have a favorite film?"
    )
    memo.add_memory(
        user_input="I enjoy painting landscapes in my free time.",
        agent_response="Painting is such a creative hobby! Do you use oils, acrylics, or watercolors?"
    )

     
    test_query = "What do you remember about my job?"
    print(f"User: {test_query}")
    
    response = memo.get_response(
        query=test_query,
    )
    
    print(f"Assistant: {response}")

if __name__ == "__main__":
    simple_demo()