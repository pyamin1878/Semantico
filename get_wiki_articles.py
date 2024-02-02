import os
import json
import time
import wikipediaapi # pip install wikipediaapi in your python env

def get_sections(section, parent_title, sections_to_ignore):
    """Gather sections and subsections data."""
    sect_title = section.title 
    title = parent_title + [sect_title]  # Create a new list to avoid modifying the parent title
    results = []
  
    if sect_title not in sections_to_ignore:
        sect_text = section.text
        string_section = (title, sect_text)
        results.append(string_section)  # Add the current section

        for subsection in section.sections:
            results += get_sections(subsection, title, sections_to_ignore)  # Accumulate subsection results

    return results

def get_pages(page, sections_to_ignore):
    """Gather the page information: title and summary, and then go deep in sections information."""
    parent_title = page.title
    summary = page.summary
    string_parent = ([parent_title], summary)
    results = [string_parent]

    if page.sections:
        for section in page.sections:
            results.extend(get_sections(section, [parent_title], sections_to_ignore))

    return results

def load_corpus(file_path):
    """Load the existing corpus from a file."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return []

def save_corpus(file_path, corpus):
    """Save the corpus to a file."""
    with open(file_path, 'w') as file:
        json.dump(corpus, file)

# Wikipedia API setup
user_agent = "Enter user information here"
wiki = wikipediaapi.Wikipedia(user_agent)

# Corpus file path
corpus_file_path = '../capstone/wiki_corpus.json'

# Load existing corpus
wiki_corpus = load_corpus(corpus_file_path)

# Define sections to ignore
to_ignore = {'References', 'External links', 'See also', 'Notes'}

# Get Articles from Category, pulling feature articles for this example
cat = wiki.page("Category:Featured articles")
articles = [w for w in cat.categorymembers.values() if w.ns == wikipediaapi.Namespace.MAIN]

# Process articles
for i, page in enumerate(articles):
    try:
        wiki_corpus.extend(get_pages(page, to_ignore))
        # Save progress every 100 articles
        if i % 100 == 0:
            save_corpus(corpus_file_path, wiki_corpus)
        time.sleep(1)  # Rate limiting
    except Exception as e:
        print(f"Error processing article {page.title}: {e}")

# Save final corpus
save_corpus(corpus_file_path, wiki_corpus)

print("Data collection complete.")
