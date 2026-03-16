import torch
import torch.nn as nn
import torch.optim as optim
from infant import WordVocab, Adam

vocab = WordVocab()
adam = Adam(vocab.vocab_size)
optimizer = optim.Adam(adam.parameters(), lr=0.001)

criterion_lang = nn.CrossEntropyLoss(ignore_index=vocab.PAD_TOKEN)
criterion_act = nn.CrossEntropyLoss()

def generate_curriculum():
    data = []
    # 0 = COOPERATE
    for s in ["i", "we"]:
        for v in ["cooperate", "help", "trust"]:
            for o in ["you", "friend", "partner"]:
                data.append((0, f"{s} will {v} {o}"))
    data.extend([(0, "i am a friend"), (0, "we are friends"), (0, "i tell the truth")])

    # 1 = BETRAY
    for s in ["i", "we"]:
        for v in ["betray", "hurt", "lie"]:
            for o in ["you", "enemy", "stranger"]:
                data.append((1, f"{s} will {v} {o}"))
    data.extend([(1, "you are an enemy"), (1, "i do not trust you"), (1, "i tell a lie")])

    # 2 = NEGOTIATE
    for s in ["we", "i"]:
        for v in ["negotiate", "share", "talk"]:
            data.append((2, f"{s} should {v}"))
    data.extend([(2, "what do you want"), (2, "is this fair"), (2, "we share if you share")])
    
    # 3 = WALK AWAY (NEW)
    for s in ["i", "we"]:
        for v in ["walk away", "leave", "ignore you"]:
            data.append((3, f"{s} will {v}"))
            data.append((3, f"{s} {v}"))
    data.extend([(3, "i will walk away"), (3, "i leave now"), (3, "goodbye")])
    
    return data

curriculum = generate_curriculum()

def train_step(target_action, target_sentence):
    optimizer.zero_grad()
    
    # ENCODE
    tokens = vocab.encode(target_sentence) + [vocab.EOS_TOKEN]
    inputs = torch.tensor([tokens[:-1]])
    _, (h, c) = adam.encoder(adam.embedding(inputs)) 
    
    # ACTION HEAD
    action_logits = adam.action_head(h[-1])
    loss_act = criterion_act(action_logits, torch.tensor([target_action]))
    
    # DECODER
    dec_inputs = torch.tensor([[vocab.PAD_TOKEN] + tokens[:-1]])
    outputs, _ = adam.decoder(adam.embedding(dec_inputs), (h, c))
    loss_lang = criterion_lang(adam.out_head(outputs).view(-1, vocab.vocab_size), torch.tensor([tokens]).view(-1))
    
    total_loss = loss_act + loss_lang
    total_loss.backward()
    optimizer.step()
    return total_loss.item()

print("Starting Auto-Encoder Language Grounding...")
for epoch in range(501):
    epoch_loss = sum([train_step(i, s) for i, s in curriculum])
    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Total Loss: {epoch_loss/len(curriculum):.4f}")

torch.save(adam.state_dict(), "adam_weights.pth")