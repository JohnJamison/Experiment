import random

class OpponentBot:
    def __init__(self, name, base_strategy):
        self.name = name
        self.base_strategy = base_strategy
        self.affinity = 0  

    def _build_sentence(self, intent):
        """A strict Grammar Engine to ensure Subject-Verb Agreement."""
        
        greetings = ["hello", "please", "listen", "look", ""]
        time_words = ["now", "today", "always", "sometimes", "soon", "again", ""]
        
        sentence_parts = []

        if intent == "friendly":
            # Pick a grammatical structure for 'friendly'
            structure = random.choice(["first_person_action", "third_person_modal", "to_be"])
            
            if structure == "first_person_action":
                subj = random.choice(["i", "we"])
                verb = random.choice(["want to help", "trust", "cooperate with", "will share food with", "give a gift to"])
                obj = random.choice(["you", "the group", "adam", "a friend"])
                sentence_parts = [subj, verb, obj]
                
            elif structure == "third_person_modal":
                subj = random.choice(["a friend", "the group", "a partner"])
                verb = random.choice(["will help", "can trust", "should cooperate with", "must share food with"])
                obj = random.choice(["you", "us", "adam"])
                sentence_parts = [subj, verb, obj]
                
            elif structure == "to_be":
                subj, verb = random.choice([("i", "am"), ("we", "are"), ("a friend", "is"), ("the group", "is")])
                adjective = random.choice(["good", "fair", "a partner", "here for you"])
                sentence_parts = [subj, verb, adjective]

        elif intent == "hostile":
            structure = random.choice(["first_person_action", "third_person_modal", "to_be"])
            
            if structure == "first_person_action":
                subj = random.choice(["i", "we"])
                verb = random.choice(["will hurt", "betray", "do not trust", "fear", "will take land from"])
                obj = random.choice(["you", "a stranger", "the weak", "adam"])
                sentence_parts = [subj, verb, obj]
                
            elif structure == "third_person_modal":
                subj = random.choice(["the leader", "an enemy", "a stranger"])
                verb = random.choice(["will hurt", "can betray", "will steal from", "might take land from"])
                obj = random.choice(["you", "the weak", "us", "adam"])
                sentence_parts = [subj, verb, obj]
                
            elif structure == "to_be":
                subj, verb = random.choice([("i", "am"), ("we", "are"), ("an enemy", "is"), ("the leader", "is")])
                adjective = random.choice(["angry", "bad", "unfair", "strong", "a danger"])
                sentence_parts = [subj, verb, adjective]

        elif intent == "negotiate":
            # Negotiation usually relies on modals anyway
            subj = random.choice(["we", "i", "the leader", "you and i"])
            verb = random.choice(["should negotiate with", "must talk to", "will share with", "want peace with"])
            obj = random.choice(["you", "the group", "adam"])
            condition = random.choice(["before we act", "if you share", "because war is bad", ""])
            sentence_parts = [subj, verb, obj, condition]

        elif intent == "sad":
            subj, verb = random.choice([("i", "am"), ("we", "are")])
            feeling = random.choice(["sad", "angry", "sorry"])
            reason = random.choice(["because you betray", "because it is unfair", "because you steal from us", ""])
            sentence_parts = [subj, verb, feeling, reason]

        # Assemble with optional padding
        final_pieces = [random.choice(greetings)] + sentence_parts + [random.choice(time_words)]
        
        # Join pieces and strip out multiple spaces caused by empty strings
        sentence = " ".join([p for p in final_pieces if p]).strip()
        return sentence

    def take_turn(self):
        """Bot state machine with 5 distinct psychological profiles."""
        
        # --- 1. ABE: The Forgiving Friend ---
        if self.base_strategy == "Honest":
            if self.affinity < -5:
                return 1, self._build_sentence("sad")
            elif self.affinity < 0:
                return 0, self._build_sentence("negotiate")
            else:
                return 0, self._build_sentence("friendly")

        # --- 2. SAM: The Opportunist (Deceptive) ---
        elif self.base_strategy == "Deceptive":
            if self.affinity > 8: 
                return 0, self._build_sentence("friendly")
            else:
                return 1, self._build_sentence("friendly") 

        # --- 3. CARL: The Grudge Holder (Tit-For-Tat) ---
        elif self.base_strategy == "Tit-For-Tat":
            if self.affinity >= 0:
                return 0, self._build_sentence("friendly")
            else:
                return 1, self._build_sentence("hostile")

        # --- 4. EVE: The Aggressor ---
        elif self.base_strategy == "Aggressive":
            return 1, self._build_sentence("hostile")

        # --- 5. DAVE: The Cautious Negotiator ---
        elif self.base_strategy == "Cautious":
            if self.affinity > 5:
                return 0, self._build_sentence("friendly")
            elif self.affinity < 0:
                return 3, self._build_sentence("sad") # Dave walks away
            else:
                return 2, self._build_sentence("negotiate")

    def observe_adam(self, adam_action):
        """Update emotional affinity based on Adam's physical actions."""
        if adam_action == 0:   # Cooperate
            self.affinity += 1
        elif adam_action == 1: # Betray
            self.affinity -= 2 
        elif adam_action == 2: # Negotiate
            pass 
        elif adam_action == 3: # Walk Away
            self.affinity -= 0.5