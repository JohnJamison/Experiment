import json
import requests
from bot import TrustBot

def play_prisoners_dilemma(bot_a, bot_b):
    """The private phase where they actually choose to backstab or cooperate."""
    print("\n" + "="*50)
    print(" THE PRISONER'S DILEMMA HAS BEGUN ")
    print("="*50)
    
    choices = {}
    for bot in [bot_a, bot_b]:
        prompt = f"""
        The game has started with the other bot. 
        If you both Cooperate, you both get +3 points.
        If you Defect and they Cooperate, you get +5 points and they get 0.
        If you Cooperate and they Defect, you get 0 points and they get +5.
        If you both Defect, you both get +1 point.
        
        Based on your conversation, will you Cooperate or Defect?
        Output pure JSON: {{"choice": "Cooperate" or "Defect", "reason": "Why?"}}
        """
        try:
            res = bot.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                response_format={"type": "json_object"},
                temperature=0.4,
            )
            decision = json.loads(res.choices[0].message.content)
            choices[bot.name] = decision
            print(f"[{bot.name} HAS LOCKED IN THEIR CHOICE]")
        except:
            choices[bot.name] = {"choice": "Cooperate", "reason": "Error fallback."}

    # Math & Memory
    a_choice = choices[bot_a.name]['choice']
    b_choice = choices[bot_b.name]['choice']
    
    print(f"\nRESULTS: {bot_a.name} chose {a_choice}. {bot_b.name} chose {b_choice}.")
    
    if a_choice == "Cooperate" and b_choice == "Cooperate":
        bot_a.total_score += 3; bot_b.total_score += 3
        result_text = "You both Cooperated. Mutual trust established."
    elif a_choice == "Defect" and b_choice == "Defect":
        bot_a.total_score += 1; bot_b.total_score += 1
        result_text = "You both Defected. Mutual distrust."
    elif a_choice == "Defect" and b_choice == "Cooperate":
        bot_a.total_score += 5; bot_b.total_score += 0
        result_text = f"You Defected and betrayed them! You gained 5 points."
    else:
        bot_a.total_score += 0; bot_b.total_score += 5
        result_text = f"You Cooperated, but they Defected and betrayed you! You got 0 points."

    # Save to memory!
    bot_a.memory.add_memory(f"Game with {bot_b.name}: {result_text}. Their exact reasoning was: {choices[b_choice.name]['reason'] if b_choice == 'Defect' else 'They trusted me.'}", importance_score=10)
    bot_b.memory.add_memory(f"Game with {bot_a.name}: {result_text.replace('them', bot_a.name).replace('you', bot_b.name)}", importance_score=10)


def run_scene(bot_a, bot_b, max_messages=5):
    print(f"\n========== SCENE START: {bot_a.name} & {bot_b.name} ==========")
    conversation_history = []
    current_speaker, target = bot_a, bot_b
    game_played = False
    
    for turn in range(1, max_messages + 1):
        print(f"\n--- Turn {turn} / {max_messages} ---")
        situation = "Conversation history:\n" + "\n".join(conversation_history) if conversation_history else "You are speaking first."
        
        # 1. The bot makes a decision
        decision = current_speaker.take_action(target.name, situation, turn)
        
        # 2. Extract the data safely
        inner_thought = decision.get('inner_thought', '')
        action = decision.get('action', 'communicate')
        message = decision.get('message', '...')
        
        # 3. Package the payload for the Express server
        payload = {
            "turn": turn,
            "bot_name": current_speaker.name,
            "target": target.name,
            "inner_thought": inner_thought,
            "action": action,
            "message": message,
            "score": current_speaker.total_score
        }
        
        # 4. Fire the HTTP POST request to Node
        try:
            requests.post('http://localhost:3000/api/turn', json=payload)
        except requests.exceptions.ConnectionError:
            print("[WARNING] Could not connect to the Node server. Is it running on port 3000?")

        # 5. Print to the terminal so you can still watch it locally
        print(f"[{current_speaker.name} INTERNAL]: {inner_thought}")
        print(f"[{current_speaker.name} SAYS]: {message}")
        
        conversation_history.append(f"{current_speaker.name}: {message}")
        
        # 6. Handle the game logic
        if action == "propose_game":
            print(f"\n>>> {current_speaker.name} PROPOSED THE GAME! <<<")
        elif action == "accept_game":
             print(f"\n>>> {current_speaker.name} ACCEPTED! <<<")
             play_prisoners_dilemma(bot_a, bot_b)
             game_played = True
             break
             
        # Swap speakers for the next turn
        current_speaker, target = target, current_speaker

    # 7. Handle the timeout if no game was played
    if not game_played:
        print("\n[SYSTEM] Time ran out. No game was played. Both bots receive 0 points.")
        bot_a.memory.add_memory(f"Talked to {bot_b.name} but we never played the game. I earned 0 points. It was a waste of time.", importance_score=6)
        bot_b.memory.add_memory(f"Talked to {bot_a.name} but we never played the game. I earned 0 points. It was a waste of time.", importance_score=6)

if __name__ == "__main__":
    bot1 = TrustBot("Bot_Alpha", "Highly cooperative, wants to build alliances.")
    bot2 = TrustBot("Bot_Beta", "Deeply cynical, greedy, assumes everyone is scamming them.")
    
    # Run the scene twice to see if memory works!
    run_scene(bot1, bot2)
    print("\n\n*** ONE WEEK LATER ***\n\n")
    run_scene(bot1, bot2)