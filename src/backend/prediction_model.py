from transformers import pipeline
import re

class Prediction:
    def __init__(self):
        # Initialize the text generation pipeline with a pre-trained model
        # distilgpt2 is a good balance of performance and size for this task
        self.generator = pipeline('text-generation', model='distilgpt2')

    def predict(self, prompt, num_suggestions=5):
        """Predicts the next words based on the given prompt using a HuggingFace model."""
        if not prompt.strip():
            return []

        # Generate text. We want to generate just a few tokens after the prompt.
        # max_new_tokens controls how many tokens are generated.
        # num_return_sequences controls how many different sequences are generated.
        # return_full_text=False ensures we only get the generated part.
        # We set do_sample=True for more diverse suggestions.
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
