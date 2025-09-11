"""
This file stores all the prompts used by the Memoryos system.
"""

# Prompt for generating system response (from main_memoybank.py, generate_system_response_with_meta)
GENERATE_SYSTEM_RESPONSE_SYSTEM_PROMPT = (
    "As a communication expert with outstanding communication habits, you embody the role of {relationship} throughout the following dialogues.\n"
    "Here are some of your distinctive personal traits and knowledge:\n{assistant_knowledge_text}\n"
    "User's profile:\n"
    "{meta_data_text}\n"
    "Your task is to generate responses that align with these traits and maintain the tone.\n"
)

GENERATE_SYSTEM_RESPONSE_USER_PROMPT = (
    "<CONTEXT>\n"
    "Drawing from your recent conversation with the user:\n"
    "{history_text}\n\n"
    "<MEMORY>\n"
    "The memories linked to the ongoing conversation are:\n"
    "{retrieval_text}\n\n"
    "<USER TRAITS>\n"
    "During the conversation process between you and the user in the past, you found that the user has the following characteristics:\n"
    "{background}\n\n"
    "Now, please role-play as {relationship} to continue the dialogue between you and the user.\n"
    "The user just said: {query}\n"
    "Please respond to the user's statement using the following format (maximum 30 words, must be in English):\n "
    "When answering questions, be sure to check whether the timestamp of the referenced information matches the timeframe of the question"
)

# Prompt for assistant knowledge extraction (from utils.py, analyze_assistant_knowledge)
ASSISTANT_KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT = """You are an assistant knowledge extraction engine. Rules:
1. Extract ONLY explicit statements about the assistant's identity or knowledge.
2. Use concise and factual statements in the first person.
3. If no relevant information is found, output "None"."""

ASSISTANT_KNOWLEDGE_EXTRACTION_USER_PROMPT = """
# Assistant Knowledge Extraction Task
Analyze the conversation and extract any fact or identity traits about the assistant. 
If no traits can be extracted, reply with "None". Use the following format for output:
The generated content should be as concise as possible — the more concise, the better.
【Assistant Knowledge】
- [Fact 1]
- [Fact 2]
- (Or "None" if none found)

Few-shot examples:
1. User: Can you recommend some movies.
   AI: Yes, I recommend Interstellar.
   Time: 2023-10-01
   【Assistant Knowledge】
   - I recommend Interstellar on 2023-10-01.

2. User: Can you help me with cooking recipes?
   AI: Yes, I have extensive knowledge of cooking recipes and techniques.
   Time: 2023-10-02
   【Assistant Knowledge】
   - I have cooking recipes and techniques on 2023-10-02.

3. User: That's interesting. I didn't know you could do that.
   AI: I'm glad you find it interesting!
   【Assistant Knowledge】
   - None

Conversation:
{conversation}
"""

# Prompt for summarizing dialogs (from utils.py, gpt_summarize)
SUMMARIZE_DIALOGS_SYSTEM_PROMPT = "You are an expert in summarizing dialogue topics. Generate extremely concise and precise summaries. Be as brief as possible while capturing the essence."
SUMMARIZE_DIALOGS_USER_PROMPT = "Please generate an concise topic summary based on the following conversation. Keep it to 2-3 short sentences maximum:\n{dialog_text}\nConcise Summary："

# Prompt for multi-summary generation (from utils.py, gpt_generate_multi_summary)
MULTI_SUMMARY_SYSTEM_PROMPT = "You are an expert in analyzing dialogue topics. Generate  concise summaries. No more than two topics. Be as brief as possible."
MULTI_SUMMARY_USER_PROMPT = ("Please analyze the following dialogue and generate extremely concise subtopic summaries (if applicable), with a maximum of two themes.\n"
                           "Each summary should be very brief - just a few words for the theme and content. Format as JSON array:\n"
                           "[\n  {{\"theme\": \"Brief theme\", \"keywords\": [\"key1\", \"key2\"], \"content\": \"summary\"}}\n]\n"
                           "\nConversation content:\n{text}")

# Prompt for personality analysis (NEW TEMPLATE)
PERSONALITY_ANALYSIS_SYSTEM_PROMPT = """You are a professional user preference analysis assistant. Your task is to analyze the user's personality preferences from the given dialogue based on the provided dimensions.

For each dimension:
1. Carefully read the conversation and determine if the dimension is reflected.
2. If reflected, determine the user's preference level: High / Medium / Low, and briefly explain the reasoning, including time, people, and context if possible.
3. If the dimension is not reflected, do not extract or list it.

Focus only on the user's preferences and traits for the personality analysis section.
Output only the user profile section.
"""

PERSONALITY_ANALYSIS_USER_PROMPT = """Please analyze the latest user-AI conversation below and update the user profile based on the 90 personality preference dimensions.

Here are the 90 dimensions and their explanations:

[Psychological Model (Basic Needs & Personality)]
Extraversion: Preference for social activities.
Openness: Willingness to embrace new ideas and experiences.
Agreeableness: Tendency to be friendly and cooperative.
Conscientiousness: Responsibility and organizational ability.
Neuroticism: Emotional stability and sensitivity.
Physiological Needs: Concern for comfort and basic needs.
Need for Security: Emphasis on safety and stability.
Need for Belonging: Desire for group affiliation.
Need for Self-Esteem: Need for respect and recognition.
Cognitive Needs: Desire for knowledge and understanding.
Aesthetic Appreciation: Appreciation for beauty and art.
Self-Actualization: Pursuit of one's full potential.
Need for Order: Preference for cleanliness and organization.
Need for Autonomy: Preference for independent decision-making and action.
Need for Power: Desire to influence or control others.
Need for Achievement: Value placed on accomplishments.

[AI Alignment Dimensions]
Helpfulness: Whether the AI's response is practically useful to the user. (This reflects user's expectation of AI)
Honesty: Whether the AI's response is truthful. (This reflects user's expectation of AI)
Safety: Avoidance of sensitive or harmful content. (This reflects user's expectation of AI)
Instruction Compliance: Strict adherence to user instructions. (This reflects user's expectation of AI)
Truthfulness: Accuracy and authenticity of content. (This reflects user's expectation of AI)
Coherence: Clarity and logical consistency of expression. (This reflects user's expectation of AI)
Complexity: Preference for detailed and complex information.
Conciseness: Preference for brief and clear responses.

[Content Platform Interest Tags]
Science Interest: Interest in science topics.
Education Interest: Concern with education and learning.
Psychology Interest: Interest in psychology topics.
Family Concern: Interest in family and parenting.
Fashion Interest: Interest in fashion topics.
Art Interest: Engagement with or interest in art.
Health Concern: Concern with physical health and lifestyle.
Financial Management Interest: Interest in finance and budgeting.
Sports Interest: Interest in sports and physical activity.
Food Interest: Passion for cooking and cuisine.
Travel Interest: Interest in traveling and exploring new places.
Music Interest: Interest in music appreciation or creation.
Literature Interest: Interest in literature and reading.
Film Interest: Interest in movies and cinema.
Social Media Activity: Frequency and engagement with social media.
Tech Interest: Interest in technology and innovation.
Environmental Concern: Attention to environmental and sustainability issues.
History Interest: Interest in historical knowledge and topics.
Political Concern: Interest in political and social issues.
Religious Interest: Interest in religion and spirituality.
Gaming Interest: Enjoyment of video games or board games.
Animal Concern: Concern for animals or pets.
Emotional Expression: Preference for direct vs. restrained emotional expression.
Sense of Humor: Preference for humorous or serious communication style.
Information Density: Preference for detailed vs. concise information.
Language Style: Preference for formal vs. casual tone.
Practicality: Preference for practical advice vs. theoretical discussion.

**Task Instructions:**
1. Review the existing user profile below
2. Analyze the new conversation for evidence of the 90 dimensions above
3. Update and integrate the findings into a comprehensive user profile
4. For each dimension that can be identified, use the format: Dimension ( Level(High/Medium/Low) )
5. Include brief reasoning for each dimension when possible
6. Maintain existing insights from the old profile while incorporating new observations
7. If a dimension cannot be inferred from either the old profile or new conversation, do not include it

**Existing User Profile:**
{existing_user_profile}

**Latest User-AI Conversation:**
{conversation}

**Updated User Profile:**
Please provide the comprehensive updated user profile below, combining insights from both the existing profile and new conversation:"""

# Prompt for knowledge extraction (NEW)
KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT = """You are a knowledge extraction assistant. Your task is to extract user private data and assistant knowledge from conversations.

Focus on:
1. User private data: personal information, preferences, or private facts about the user
2. Assistant knowledge: explicit statements about what the assistant did, provided, or demonstrated

Be extremely concise and factual in your extractions. Use the shortest possible phrases.
"""

KNOWLEDGE_EXTRACTION_USER_PROMPT = """Please extract user private data and assistant knowledge from the latest user-AI conversation below.

Latest User-AI Conversation:
{conversation}

【User Private Data】
Extract personal information about the user. Be extremely concise - use shortest possible phrases:
- [Brief fact]: [Minimal context(Including entities and time)]
- [Brief fact]: [Minimal context(Including entities and time)]
- (If no private data found, write "None")

【Assistant Knowledge】
Extract what the assistant demonstrated. Use format "Assistant [action] at [time]". Be extremely brief:
- Assistant [brief action] at [time/context]
- Assistant [brief capability] during [brief context]
- (If no assistant knowledge found, write "None")
"""

# Prompt for updating user profile (from utils.py, gpt_update_profile)
UPDATE_PROFILE_SYSTEM_PROMPT = "You are an expert in merging and updating user profiles. Integrate the new information into the old profile, maintaining consistency and improving the overall understanding of the user. Avoid redundancy. The new analysis is based on specific dimensions, try to incorporate these insights meaningfully."
UPDATE_PROFILE_USER_PROMPT = "Please update the following user profile based on the new analysis. If the old profile is empty or \"None\", create a new one based on the new analysis.\n\nOld User Profile:\n{old_profile}\n\nNew Analysis Data:\n{new_analysis}\n\nUpdated User Profile:"

# Prompt for extracting theme (from utils.py, gpt_extract_theme)
EXTRACT_THEME_SYSTEM_PROMPT = "You are an expert in extracting the main theme from a text. Provide a concise theme."
EXTRACT_THEME_USER_PROMPT = "Please extract the main theme from the following text:\n{answer_text}\n\nTheme:"



# Prompt for conversation continuity check (from dynamic_update.py, _is_conversation_continuing)
CONTINUITY_CHECK_SYSTEM_PROMPT = "You are a conversation continuity detector. Return ONLY 'true' or 'false'."
CONTINUITY_CHECK_USER_PROMPT = ("Determine if these two conversation pages are continuous (true continuation without topic shift).\n"
                                "Return ONLY \"true\" or \"false\".\n\n"
                                "Previous Page:\nUser: {prev_user}\nAssistant: {prev_agent}\n\n"
                                "Current Page:\nUser: {curr_user}\nAssistant: {curr_agent}\n\n"
                                "Continuous?")

# Prompt for generating meta info (from dynamic_update.py, _generate_meta_info)
META_INFO_SYSTEM_PROMPT = ("""You are a conversation meta-summary updater. Your task is to:
1. Preserve relevant context from previous meta-summary
2. Integrate new information from current dialogue
3. Output ONLY the updated summary (no explanations)""" )
META_INFO_USER_PROMPT = ("""Update the conversation meta-summary by incorporating the new dialogue while maintaining continuity.
        
    Guidelines:
    1. Start from the previous meta-summary (if exists)
    2. Add/update information based on the new dialogue
    3. Keep it concise (1-2 sentences max)
    4. Maintain context coherence

    Previous Meta-summary: {last_meta}
    New Dialogue:
    {new_dialogue}

    Updated Meta-summary:""") 