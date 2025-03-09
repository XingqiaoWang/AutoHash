from lxml import etree
from FlagEmbedding import BGEM3FlagModel
import os
import numpy as np

def embedding(data_set):
    Embedding_model = BGEM3FlagModel('BAAI/bge-m3',
                        use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

    record_outputs = Embedding_model.encode(data_set,
                                batch_size=196,
                                max_length=1024, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                return_dense=True,
                                return_sparse=True,
                                )
    return record_outputs['dense_vecs']

def extract_all_papers_as_strings(file_path, max_records=None):
    """Extract all DBLP papers and format them as concatenated strings."""
    parser = etree.XMLParser(recover=True, resolve_entities=True)
    tree = etree.parse(file_path, parser)
    root = tree.getroot()

    paper_strings = []

    for i, elem in enumerate(root.findall("article") + root.findall("inproceedings")):
        title = elem.find("title")
        authors = [author.text if author is not None else "Unknown" for author in elem.findall("author")]
        year = elem.find("year")
        venue = elem.find("journal") if elem.tag == "article" else elem.find("booktitle")
        doi = elem.find("ee")
        pages = elem.find("pages")
        crossref = elem.find("crossref")

        # Ensure all extracted attributes are safe to use
        title_text = title.text if title is not None else "Unknown Title"
        year_text = year.text if year is not None else "Unknown Year"
        venue_text = venue.text if venue is not None else "Unknown Venue"
        doi_text = doi.text if doi is not None else "No DOI"
        pages_text = pages.text if pages is not None else "Unknown Pages"
        crossref_text = crossref.text if crossref is not None else "No References"

        # Convert all authors to strings, ensuring no NoneType values exist
        authors_text = ', '.join(str(author) for author in authors) if authors else "Unknown Author"

        # Concatenate all information into a formatted string
        paper_string = f"Title: {title_text}. " \
                       f"Authors: {authors_text}. " \
                       f"Year: {year_text}. " \
                       f"Venue: {venue_text}. " \
                       f"DOI: {doi_text}. " \
                       f"Pages: {pages_text}. " \
                       f"References: {crossref_text}."

        paper_strings.append(paper_string)

        if max_records and i >= max_records:
            break  # Limit the number of records for testing

    return paper_strings

def save_embeddings_in_chunks(embeddings, output_dir, chunk_size=1000000):
    """
    Saves a NumPy array of embeddings into multiple .npy files.

    Args:
        embeddings: The NumPy array of embeddings.
        output_dir: The directory to save the .npy files.
        chunk_size: The number of embeddings to save in each file.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_embeddings = len(embeddings)
    print(len(embeddings))
    num_chunks = (num_embeddings + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_embeddings)
        chunk = embeddings[start:end]

        output_file = os.path.join(output_dir, f"embeddings_chunk_{i}.npy")
        np.save(output_file, chunk)
        print(f"Saved chunk {i + 1} of {num_chunks} to {output_file}")

# Load DBLP dataset (Limit to 50,000 records for testing)
dblp_file = "dblp.xml"
papers = extract_all_papers_as_strings(dblp_file)
emb = embedding(papers)


# Save embeddings
output_dir = '/scrfs/storage/xwang1/home/pseudopeople_dataset/workspace/embedding_indexing/model_parameter_evaluation/dblp/dblp_embedding_chunks'
save_embeddings_in_chunks(emb,output_dir)






