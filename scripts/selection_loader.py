import wget
import os
from dotenv import load_dotenv
load_dotenv()


BIN_FOLDER = 'D:/PREreviewBOT/bin'

def link_to_pdf(insert_link):
    """ 
    This function downloads the file from links that contain strings like 'arXiv', 'bioRxiv',
    'medRxiv', 'ChemRxiv', or 'Zenodo'. If such strings are present, it downloads and returns 
    the PDF in the variable 'document'.
    """
    
    # Allowed keywords (case-insensitive matching)
    keywords = ['arXiv', 'bioRxiv', 'medRxiv', 'ChemRxiv', 'Zenodo']
    
    # Convert the link to lowercase for case-insensitive matching
    lower_link = insert_link.lower()
    
    # Check if the link contains any of the allowed keywords
    if any(keyword.lower() in lower_link for keyword in keywords):
        try:
            # Download the file using wget
            file_path = wget.download(insert_link)
            
            # Read the file as bytes
            with open(file_path, 'rb') as file:
                document = file.read()
            
            # Optionally, remove the file after reading it
            os.remove(file_path)
            
            return document
        
        except Exception as e:
            print(f"Error downloading the file: {e}")
            return None
    else:
        print(f"Invalid link. Only links containing one of the allowed sources {keywords} are accepted.")
        return None


