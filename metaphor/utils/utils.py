import torch
import pickle


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


def forward(sentences, model, tokenizer, embedding, batch_size=128, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logits = torch.zeros((len(sentences), 3))
    tokenized_sentences = tokenizer(sentences)

    model.to(device)
    embedding.to(device)
    tokenized_sentences = tokenized_sentences.to(device)
    embedding = embedding(tokenized_sentences).to(device)

    for start in range(0, len(sentences), batch_size):
        end = min(len(sentences), start + batch_size)
        y = model(embedding[start:end], torch.arange(start, end).to(device))
        logits[start:end] = torch.softmax(y, dim=1)

    return logits


def predict(sentences, model, tokenizer, embedding, batch_size=128, device=None):
    return forward(sentences, model, tokenizer, embedding, batch_size, device).argmax(
        dim=1
    )
