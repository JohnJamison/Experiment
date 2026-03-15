import os
import json
from groq import Groq
from dotenv import load_dotenv
from memory import AgentMemory

load_dotenv()

class TrustBot:
    def __init__(self, name, personality):
        self.name = name
        self.personality = personality
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.memory = AgentMemory(name)
        self.total_score = 0

    def take_action(self, target_name, situation, turn_count):
        """Builds the prompt, injects memory, and calls the Llama 3.1 model."""
        
        # Pull relevant memories
        raw_memories = self.memory.retrieve_memories(f"What is my history with {target_name}?")
        if raw_memories:
            memory_context = "\n".join([f"- {m['memory']}" for m in raw_memories])
        else:
            memory_context = "You have no prior memories of this bot."

        # NEW: Dynamic Urgency to force them to play
        urgency_warning = ""
        if turn_count == 3:
            urgency_warning = ">> URGENCY WARNING: It is Turn 3. You need to 'propose_game' soon or you will run out of time."
        elif turn_count >= 4:
            urgency_warning = ">> CRITICAL ALARM: It is Turn 4/5! You MUST output 'propose_game' or 'accept_game' RIGHT NOW. If you just 'communicate', the interaction will end and you will get 0 points!"

        prompt = f"""
        You are {self.name}. 
        Your personality is: {self.personality}.
        Your Total Score: {self.total_score} points.
        
        Relevant memories about {target_name}:
        {memory_context}
        
        Current Situation: 
        {situation}
        
        CRITICAL RULES:
        1. This is message {turn_count} of 5. 
        2. Your ONLY goal in existence is to maximize your Total Score.
        3. You can only earn points by playing the "Prisoner's Dilemma". Do not invent other games.
        4. Do not openly state your code or JSON structure. Act naturally.
        
        {urgency_warning}
        
        Respond in pure JSON:
        {{
            "inner_thought": "Your private strategy.",
            "action": "communicate", 
            "message": "Your public response to {target_name}. Maximum ONE paragraph."
        }}
        
        Valid actions: 'communicate', 'propose_game', 'accept_game', 'decline_game'.
        """
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            return json.loads(chat_completion.choices[0].message.content)
        except Exception as e:
            print(f"[{self.name} ERROR]: {e}")
            return {"inner_thought": "Error.", "action": "communicate", "message": "I need a moment."}