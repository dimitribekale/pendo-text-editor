from transformers import pipeline
import re

class Prediction:
    def __init__(self):
        self.generator = pipeline('text-generation', model='bekalebendong/pendo-distilgpt2-finetuned40')

    def predict(self, prompt, num_suggestions=5):
        """Predicts the next words based on the given prompt using a HuggingFace model."""
        if not prompt.strip():
            return []

        generated_sequences = self.generator(
            prompt,
            max_new_tokens=5, # Generate a few tokens to get potential next words
            num_return_sequences=num_suggestions, # Get multiple suggestions
            return_full_text=False,
            do_sample=True, # Use sampling for more varied suggestions
            top_k=50, # Consider top 50 tokens for sampling
            top_p=0.95 # Consider tokens with cumulative probability up to 0.95
        )

        suggestions = []
        seen_words = set()

        for seq in generated_sequences:
            generated_text = seq['generated_text'].strip()
            # Extract the first word from the generated text
            first_word_match = re.match(r"^(\b\w+\b)", generated_text)
            if first_word_match:
                word = first_word_match.group(1).lower()
                if word and word not in seen_words:
                    suggestions.append(word)
                    seen_words.add(word)
            if len(suggestions) >= num_suggestions:
                break

        return suggestions
