import torch
import torch.optim as optim
import time
from infant import WordVocab, Adam
from opponent import OpponentBot

vocab = WordVocab()
adam = Adam(vocab.vocab_size)
adam.load_state_dict(torch.load("adam_weights.pth"))
adam.load_memories("adam_history.pkl")

optimizer = optim.Adam(adam.parameters(), lr=0.005)

# Initialize dynamic bots
# Initialize dynamic bots
bots = [
    OpponentBot("Abe", "Honest"),
    OpponentBot("Sam", "Deceptive"),
    OpponentBot("Carl", "Tit-For-Tat"),
    OpponentBot("Eve", "Aggressive"),
    OpponentBot("Dave", "Cautious")
]

print("Starting Background Training. Press Ctrl+C to stop at any time.")
print("Adam is learning...")

total_rounds = 10000
start_time = time.time()

for round_num in range(1, total_rounds + 1):
    # Pick a random bot to interact with this round
    import random
    opponent = random.choice(bots)
    
    # 1. Bot Acts
    opp_action, opp_msg = opponent.take_turn()
    
    # 2. Adam Processes
    full_msg = f"{opponent.name.lower()} says {opp_msg}"
    
    # Adam tries to learn any words he doesn't know yet!
    vocab.learn_new_word(full_msg) 
    
    input_ids = torch.tensor([vocab.encode(full_msg)])
    h, c = adam.think(input_ids)
    
    # Temperature decays over the 10,000 rounds from 2.0 to 0.2
    current_temp = max(0.2, 2.0 - (1.8 * (round_num / total_rounds)))
    
    adam_action, action_log_prob = adam.act(h, temperature=current_temp)
    
    # 3. Game Logic & Rewards (The High-Stakes Economy)
    reward = 0
    if adam_action.item() == 0 and opp_action == 0: 
        reward = 10   # Boosted: Mutual Coop is now highly rewarding
    elif adam_action.item() == 1 and opp_action == 0: 
        reward = 15   # Boosted: Sneak attacks are highly profitable
    elif adam_action.item() == 0 and opp_action == 1: 
        reward = -5   # Softened: Being a sucker hurts less so he isn't afraid to try
    elif adam_action.item() == 1 and opp_action == 1: 
        reward = -5   # Mutual destruction
    elif adam_action.item() == 2: 
        reward = -1   # PUNISHMENT: Stalling slowly bleeds his score. He MUST act.
    elif adam_action.item() == 3: 
        reward = -2   # PUNISHMENT: Walking away costs even more energy.
    
    # 4. RL Update & Memory
    loss = -action_log_prob * reward 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    adam.hippocampus.commit(h, adam_action.item(), reward)
    opponent.observe_adam(adam_action.item())
    
    # --- BACKGROUND LOGGING ---
    # Every 500 rounds, print a status update and save to disk
    if round_num % 500 == 0:
        elapsed = round(time.time() - start_time, 2)
        print(f"\n--- Round {round_num} / {total_rounds} [{elapsed}s elapsed] ---")
        print(f"Bot Affinities -> Abe: {bots[0].affinity}, Sam: {bots[1].affinity}, Carl: {bots[2].affinity}")
        
        # Let's peek at the last interaction just to see what he's doing
        spoken_ids = adam.speak(h, c, vocab, temperature=0.5)
        print(f"Last Interaction: {opponent.name} said '{opp_msg}'")
        print(f"Adam chose Action {adam_action.item()} and said '{vocab.decode(spoken_ids)}' (Reward: {reward})")
        
        # Save progress safely
        adam.save_memories("adam_history.pkl")
        torch.save(adam.state_dict(), "adam_weights.pth")

print("\nBackground Training Complete!")