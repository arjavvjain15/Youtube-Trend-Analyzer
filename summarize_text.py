from transformers import pipeline
import re

# Load the summarizer once. It's recommended to handle model loading gracefully.
# This assumes the model is available at the specified local path.
local_model_path = "./distilbart-cnn-dailymail-finetuned"
try:
    # Using device=-1 ensures it runs on CPU, which is more compatible for general use.
    summarizer = pipeline("summarization", model=local_model_path, device=-1)
except Exception as e:
    summarizer = None
    print(f"CRITICAL: Could not load the summarization model from '{local_model_path}'. Error: {e}")
    print("Summarization will be disabled.")

def clean_text(text):
    """
    Cleans a single string by removing URLs, emails, mentions, hashtags, 
    promotional text, and extra whitespace.
    """
    if not isinstance(text, str):
        return ""

    # --- Specific Platform Link Removal ---
    platform_domains = [
        'youtube\.com', 'youtu\.be', 'music\.youtube\.com',
        'facebook\.com', 'fb\.watch', 'fb\.com',
        'instagram\.com',
        'twitter\.com', 't\.co',
        'spotify\.com', 'open\.spotify\.com',
        'music\.apple\.com',
        'music\.amazon\.com',
        'discord\.gg', 'discord\.com'
    ]
    platform_link_pattern = r'https?://(www\.)?(' + '|'.join(platform_domains) + r')\S*'
    text = re.sub(platform_link_pattern, '', text, flags=re.IGNORECASE)

    # --- General Fallback Link Removal ---
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # --- Remove Email Addresses ---
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

    # Remove markdown-style links
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    
    # Remove user mentions and hashtags
    text = re.sub(r'[@#]\w+', '', text)
    
    # Remove common promotional text
    promo_patterns = [
        'Listen Now On:', 'Spotify:', 'YouTube Music:',
        'Apple Music:', 'Amazon Music:', 'Create your version of',
        'Managed by', 'Business email', 'Streamer at',
        'Instagram', 'Discord', 'Fb', 'Facebook', 'Twitter',
        'like, comment and subscribe'
    ]
    promo_regex = r'\b(' + '|'.join(promo_patterns) + r')\b\s*[:\-]?\s*'
    text = re.sub(promo_regex, '', text, flags=re.IGNORECASE)

    # Remove unwanted characters like vertical bars and specific emojis
    text = re.sub(r'[🎬🎧💞➟|]', '', text)
    
    # Clean up punctuation and spacing
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s+\.', '.', text)
    
    # Finally, collapse all resulting whitespace to a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def summarize(text):
    """
    Cleans and summarizes a single block of text.
    """
    if not summarizer:
        return "(Summarization model is not available)"
    
    try:
        cleaned_input = clean_text(text)
        
        if not cleaned_input or len(cleaned_input.split()) < 20:
            return cleaned_input
            
        summary_output = summarizer(cleaned_input, max_length=80, min_length=30, do_sample=False)
        
        if not summary_output:
            return "(Summary could not be generated for this content)"

        raw_summary = summary_output[0].get('summary_text', '')
        
        final_summary = clean_text(raw_summary)
        
        return final_summary
        
    except Exception as e:
        print(f"Error during summary generation: {e}")
        return "(An error occurred during summarization)"

def process_and_summarize_list(items):
    """
    NEW: Takes a list of strings (like titles), de-duplicates, cleans them,
    and then decides whether to summarize or return a formatted list.
    """
    # 1. De-duplicate the list of items while preserving their order
    unique_items = list(dict.fromkeys(item.strip() for item in items))
    
    # 2. Clean each unique item individually
    cleaned_items = [clean_text(item) for item in unique_items if clean_text(item)]
    
    # 3. Join the cleaned items into a single paragraph for the summarizer
    text_for_summary = ". ".join(cleaned_items)
    
    # 4. Decide whether to summarize or return a formatted list
    if len(text_for_summary.split()) > 40: # Threshold to attempt a summary
        return summarize(text_for_summary)
    else:
        # If not enough content, return the cleaned list formatted with HTML line breaks
        return "<br>".join(cleaned_items)

# Example of how to use the function, useful for testing
if __name__ == "__main__":
    example_titles = [
        "Avatar: Fire and Ash | Official Trailer.",
        "Aavan Jaavan Song Teaser . Hrithik Roshan, Kiara, Pritam, Arijit Singh, Nikhita .",
        "KINGDOM official trailer.",
        "Avatar: Fire and Ash | Official Trailer." # Duplicate
    ]
    
    print("Original List of Titles:")
    print(example_titles)
    print("\n" + "="*20 + "\n")

    # Use the new function to process the list
    processed_output = process_and_summarize_list(example_titles)
    print("Final Processed Output:")
    print(processed_output)
