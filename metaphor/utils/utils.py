def predict(sentences, model, tokenizer, embedding, batch_size=128, device=None):
    if device is None:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
    predictions = torch.zeros((len(sentences))
    print("Finished computing attacked sentences")
    tokenized_sentences = tokenizer(sentences)
    embedding = embedding(tokenized_sentences).to(device)

    print("Attacking the model")
    for start in range(0, len(sentences), batch_size):
        end = min(len(sentences), start + batch_size)
        y = model(embedding[start:end], np.arange(start, end))
        predictions[start:end] = torch.argmax(y, dim=1)

    return predictions
