import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import pickle 
import os 

class WordVocab:
    def __init__(self):
        # Basic Structural
        self.words = ['<PAD>', '<EOS>', '<UNK>', 'i', 'you', 'he', 'she', 'we', 'they', 'it']
        self.words += ['am', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had']
        self.words += ['will', 'shall', 'should', 'would', 'can', 'could', 'may', 'might', 'must']
        self.words += ['not', 'no', 'yes', 'maybe', 'very', 'really', 'too', 'again', 'now', 'later']
        self.words += ['a', 'the', 'an', 'says'] 
        
        # Connectives & Logic
        self.words += ['and', 'but', 'or', 'if', 'then', 'because', 'so', 'although', 'unless']
        self.words += ['who', 'what', 'where', 'when', 'why', 'how', 'which', 'this', 'that']
        
        # Social, Political & IDENTITIES
        self.words += ['friend', 'enemy', 'leader', 'follower', 'stranger', 'partner', 'group', 'alliance']
        self.words += ['abe', 'sam', 'carl', 'eve', 'dave', 'mother', 'adam']
        self.words += ['power', 'weak', 'strong', 'equal', 'rule', 'law', 'fair', 'unfair', 'justice']
        self.words += ['trust', 'betray', 'cooperate', 'negotiate', 'lie', 'truth', 'promise', 'secret']
        self.words += ['share', 'keep', 'steal', 'give', 'take', 'owe', 'debt', 'pay', 'gift', 'resource']
        self.words += ['food', 'gold', 'shelter', 'land', 'border', 'war', 'peace', 'help', 'hurt']
        
        # Action 3 Vocabulary: Boundaries
        self.words += ['walk', 'away', 'leave', 'ignore', 'bye', 'done', 'stop', 'quit']
        
        # Emotional & Cognitive
        self.words += ['think', 'know', 'feel', 'want', 'hope', 'fear', 'angry', 'happy', 'sad', 'sorry']
        self.words += ['good', 'bad', 'right', 'wrong', 'better', 'worse', 'best', 'worst', 'true', 'false']
        
        # Time & Probability
        self.words += ['yesterday', 'today', 'tomorrow', 'always', 'never', 'sometimes', 'often', 'rarely']
        self.words += ['before', 'after', 'soon', 'long', 'time', 'first', 'last', 'next', 'back']
        
        # Fillers
        self.words += ['please', 'thanks', 'hello', 'goodbye', 'go', 'wait', 'look', 'listen', 'speak']
        self.words += ['one', 'many', 'all', 'some', 'none', 'more', 'less', 'big', 'small', 'here', 'there']
        # Prepositions (The glue of the English language)
        self.words += ['with', 'to', 'from', 'for', 'about', 'at', 'by', 'in', 'on', 'as', 'of']
        self.words += ['do', 'talk', 'make', 'say'] # Added a few extra useful verbs just in case

        for i in range(100):
            self.words.append(f"<BLANK_{i}>")
            
        self.word2int = {w: i for i, w in enumerate(self.words)}
        self.int2word = {i: w for i, w in enumerate(self.words)}
        self.PAD_TOKEN = self.word2int['<PAD>']
        self.EOS_TOKEN = self.word2int['<EOS>']
        self.vocab_size = len(self.words)

    def learn_new_word(self, text_string):
        """Scans a string. If it finds a new word, it claims a blank neuron for it."""
        clean = text_string.lower().replace('.', '').replace('?', '').split()
        
        for word in clean:
            if word not in self.word2int:
                # Find the first available blank slot
                for i in range(self.vocab_size):
                    if self.int2word[i].startswith("<BLANK_"):
                        old_blank = self.int2word[i]
                        
                        # Overwrite the blank slot with the new word
                        self.int2word[i] = word
                        self.word2int[word] = i
                        del self.word2int[old_blank]
                        
                        print(f"\n[NEUROPLASTICITY] Adam learned a new word: '{word}' (Wired to Neuron {i})")
                        break # Move to the next word in the string

    def encode(self, text):
        clean = text.lower().replace('.', '').replace('?', '').split()
        return [self.word2int.get(w, self.word2int['<UNK>']) for w in clean]

    def decode(self, ints):
        return " ".join([self.int2word.get(i, '') for i in ints if i not in [self.PAD_TOKEN, self.EOS_TOKEN] and not self.int2word.get(i, '').startswith("<BLANK_")])
class Hippocampus:
    def __init__(self, max_memories=500):
        self.max_memories = max_memories
        self.memories = [] 

    def commit(self, context_vec, action, reward):
        mem = {'vector': context_vec.detach().cpu().numpy(), 'action': action, 'reward': reward}
        self.memories.append(mem)
        if len(self.memories) > self.max_memories:
            self.memories.sort(key=lambda x: abs(x['reward']), reverse=True)
            self.memories.pop(-1)

    def recollect(self, current_vec):
        if not self.memories: return None
        curr = current_vec.detach().cpu().numpy().flatten()
        # Find the most similar past experience via dot product/similarity
        sims = [np.dot(curr, m['vector'].flatten()) / (np.linalg.norm(curr) * np.linalg.norm(m['vector'].flatten()) + 1e-8) for m in self.memories]
        return self.memories[np.argmax(sims)]

class Agent(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(Agent, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, 128)
        self.encoder = nn.LSTM(128, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(128, hidden_size, batch_first=True)
        self.out_head = nn.Linear(hidden_size, vocab_size)
        self.action_head = nn.Linear(hidden_size, 4) 
        self.hippocampus = Hippocampus()

    def save_memories(self, filename="adam_history.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.hippocampus.memories, f)

    def load_memories(self, filename="adam_history.pkl"):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.hippocampus.memories = pickle.load(f)
            print(f"Adam has successfully recalled {len(self.hippocampus.memories)} past experiences.")
        else:
            print("Adam has no prior memories. Today is his first day.")

    def think(self, input_ids):
        """Adam's core cognitive process: Listen, Remember, and Form a Thought."""
        _, (h, c) = self.encoder(self.embedding(input_ids))
        
        # Check for past flashbacks
        past = self.hippocampus.recollect(h)
        if past:
            past_v = torch.from_numpy(past['vector']).to(h.device)
            # Influence current thought with memory
            h = (0.9 * h) + (0.1 * past_v)
        return h, c

    def act(self, h, temperature=1.0):
        """Adam decides what to do. High temperature = taking risks/exploring."""
        logits = self.action_head(h[-1])
        # Force exploration by flattening the probabilities
        logits = logits / max(temperature, 1e-8) 
        probs = F.softmax(logits, dim=-1)
        m = Categorical(probs)
        action = m.sample()
        return action, m.log_prob(action)

    def speak(self, h, c, vocab, max_len=10, temperature=1.0):
        sentence = []
        # Start with a 'silence' token
        curr = torch.tensor([[vocab.PAD_TOKEN]]) 
        
        for _ in range(max_len):
            out, (h, c) = self.decoder(self.embedding(curr), (h, c))
            logits = self.out_head(out[:, -1, :])
            
            # --- TEMPERATURE SCALING ---
            # Flattens or sharpens the probability curve
            logits = logits / max(temperature, 1e-8) 
            
            probs = F.softmax(logits, dim=-1)
            m = Categorical(probs)
            nxt = m.sample() 
            
            if nxt.item() == vocab.EOS_TOKEN: break
            sentence.append(nxt.item())
            curr = nxt.unsqueeze(0)
            
        return sentence

class Adam(Agent):
    def __init__(self, vocab_size):
        super(Adam, self).__init__(vocab_size)
        self.name = "Adam"