import sys
sys.path.insert(0, r"d:\Abhera(Mini)")
sys.path.insert(0, r"d:\Abhera(Mini)\backend")

import torch
from ml.inference.predict_ner import NEREntityPredictor, extract_spans_from_bio

def main():
    predictor = NEREntityPredictor()
    text = "My husband keeps sending threatening messages on Instagram."
    
    inputs = predictor.tokenizer(
        text,
        truncation=True,
        max_length=256,
        return_tensors="pt",
        return_offsets_mapping=True
    )
    
    offset_mapping = inputs.pop("offset_mapping")[0].cpu().numpy()
    input_ids = inputs["input_ids"].to(predictor.device)
    attention_mask = inputs["attention_mask"].to(predictor.device)
    
    with torch.no_grad():
        outputs = predictor.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0]
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        
    tokens = predictor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    tags = [predictor.id2label.get(p, "O") for p in preds]
    
    print("TOKENIZATION & BIO TAGS:")
    for idx, (tok, tag, (start_c, end_c)) in enumerate(zip(tokens, tags, offset_mapping)):
        substr = text[start_c:end_c] if start_c != end_c else ""
        print(f"{idx:2d}: Token={tok:<15} Tag={tag:<15} Offset=({start_c},{end_c}) Substr='{substr}'")
        
    spans = extract_spans_from_bio(tags)
    print("\nEXTRACTED SPANS:", spans)
    
    print("\nCURRENT RECONSTRUCTED ENTITIES:")
    for start_idx, end_idx, label in spans:
        # Subword alignment fix: trace back to word start if starting on a subword
        s_idx = start_idx
        while s_idx > 0 and tokens[s_idx].startswith("##"):
            s_idx -= 1

        e_idx = end_idx
        while e_idx + 1 < len(tokens) and tokens[e_idx + 1].startswith("##"):
            e_idx += 1

        start_char = int(offset_mapping[s_idx][0])
        end_char = int(offset_mapping[e_idx][1])
        entity_text = text[start_char:end_char].strip()
        print(f"Label={label}, Text='{entity_text}', start_char={start_char}, end_char={end_char}")

if __name__ == "__main__":
    main()
