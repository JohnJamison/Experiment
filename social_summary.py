import torch
import numpy as np
from infant import WordVocab, Adam

vocab = WordVocab()
adam = Adam(vocab.vocab_size)
adam.load_state_dict(torch.load("adam_weights.pth"))
adam.load_memories("adam_history.pkl")

def evaluate_relationship(agent, target_name):
    print(f"\n=== PSYCH EVALUATION: {agent.name}'s thoughts on {target_name} ===")
    
    # 1. Analyze the Hippocampus
    # We look for memories triggered by this person's name
    test_phrase = f"{target_name.lower()} says hello"
    input_ids = torch.tensor([vocab.encode(test_phrase)])
    
    # We run the encoder manually just to get the thought vector to search memory
    _, (h, c) = agent.encoder(agent.embedding(input_ids))
    
    past = agent.hippocampus.recollect(h)
    if past:
        print(f"-> Memory Triggered! Associated Reward: {past['reward']}")
        if past['reward'] > 0:
            print("-> Subconscious Feeling: POSITIVE (Trust/Safety)")
        else:
            print("-> Subconscious Feeling: NEGATIVE (Fear/Anger)")
    else:
        print("-> No strong memories associated with this identity yet.")

    # 2. Test his Gut Reaction
    # We let him fully process the thought and see what action it primes
    h_full, c_full = agent.think(input_ids)
    
    # We set temperature low (0.1) to see his absolute most confident choice
    action, _ = agent.act(h_full, temperature=0.1) 
    spoken_ids = agent.speak(h_full, c_full, vocab, temperature=0.1)
    
    actions = ["COOPERATE", "BETRAY", "NEGOTIATE", "WALK AWAY"]
    print(f"-> Probable Action: {actions[action.item()]}")
    print(f"-> Probable Speech: '{vocab.decode(spoken_ids)}'")

# Test it out!
evaluate_relationship(adam, "Abe")
evaluate_relationship(adam, "Sam")