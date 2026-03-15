# The SQLite version workaround for older Linux servers
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
import math


class AgentMemory:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name=f"{agent_name}_memory")
        self.current_round = 0 

    def add_memory(self, memory_text, importance_score=5):
        """Saves a memory with its importance and the time it was created."""
        self.current_round += 1
        memory_id = f"mem_{self.current_round}"
        
        self.collection.add(
            documents=[memory_text],
            metadatas=[{"round_added": self.current_round, "importance": importance_score}],
            ids=[memory_id]
        )

    def retrieve_memories(self, query_text, top_k=3, decay_rate=0.05):
        """Finds relevant memories and scores them based on relevance and recency."""
        results = self.collection.query(query_texts=[query_text], n_results=top_k)
        
        if not results['documents'] or not results['documents'][0]:
            return []

        scored_memories = []
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            
            distance = results['distances'][0][i] if results['distances'] else 1.0
            relevance = 1.0 / (1.0 + distance) 
            importance = metadata['importance']
            time_elapsed = self.current_round - metadata['round_added']
            recency_multiplier = math.exp(-decay_rate * time_elapsed)
            
            final_score = (relevance * 0.4) + ((importance / 10) * 0.3) + (recency_multiplier * 0.3)
            
            scored_memories.append({
                "memory": doc,
                "score": round(final_score, 3)
            })
        
        scored_memories.sort(key=lambda x: x['score'], reverse=True)
        return scored_memories