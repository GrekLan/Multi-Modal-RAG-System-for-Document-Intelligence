import fitz  
from PIL import Image
import pytesseract
import io
import os
import re

class DocumentProcessor:
    """
    Enhanced document processor with improved table detection,
    semantic chunking, and better metadata extraction.
    """
    
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        
    def extract_text_chunks(self):
        """
        Extract text with semantic chunking - respects paragraphs.
        """
        chunks = []
        chunk_size = 1000
        overlap = 200
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()
            
            if not text.strip():
                continue
            
            # Split by double newlines (paragraphs) first
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            current_chunk = ""
            for para in paragraphs:
                # If adding this paragraph exceeds chunk_size, save current chunk
                if len(current_chunk) + len(para) > chunk_size and current_chunk:
                    chunks.append({
                        'type': 'text',
                        'content': current_chunk.strip(),
                        'page': page_num + 1,
                        'source': f'Page {page_num + 1}'
                    })
                    # Keep overlap
                    current_chunk = current_chunk[-overlap:] + " " + para
                else:
                    current_chunk += " " + para
            
            # Add remaining chunk
            if current_chunk.strip():
                chunks.append({
                    'type': 'text',
                    'content': current_chunk.strip(),
                    'page': page_num + 1,
                    'source': f'Page {page_num + 1}'
                })
        
        return chunks
    
    def is_table_like(self, text):
        """
        Heuristic to detect if text block looks like a table.
        """
        lines = text.strip().split('\n')
        if len(lines) < 3:
            return False
        
        # Check for common table indicators
        has_numbers = sum(1 for line in lines if re.search(r'\d+', line)) > len(lines) * 0.5
        has_aligned = sum(1 for line in lines if len(line.split()) > 2) > len(lines) * 0.6
        has_separators = any(c in text for c in ['|', '\t'])
        
        return (has_numbers and has_aligned) or has_separators
    
    def extract_tables(self):
        """
        Enhanced table extraction using layout analysis.
        """
        tables = []
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            
            blocks = page.get_text("dict")["blocks"]
            
            table_candidates = []
            for block in blocks:
                if "lines" in block:
                    block_text = ""
                    for line in block["lines"]:
                        line_text = " ".join(span["text"] for span in line["spans"])
                        block_text += line_text + "\n"
                    
                    # Check if it's table-like
                    if self.is_table_like(block_text) and len(block_text.strip()) > 50:
                        table_candidates.append({
                            'text': block_text,
                            'bbox': block.get('bbox', [])
                        })
            
            # Process table candidates
            for idx, table in enumerate(table_candidates):
                tables.append({
                    'type': 'table',
                    'content': table['text'].strip(),
                    'page': page_num + 1,
                    'source': f'Table {idx+1} on Page {page_num + 1}'
                })
        
        return tables
    
    def extract_images_with_ocr(self, output_folder=None):
        """
        Extract images and apply OCR with preprocessing.
        """
        if output_folder is None:
            try:
                import config
                output_folder = config.IMAGES_DIR
            except:
                output_folder = 'extracted_images'
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        images_data = []
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]
              
                image_filename = f"{output_folder}/page{page_num+1}_img{img_index+1}.png"
                with open(image_filename, "wb") as image_file:
                    image_file.write(image_bytes)
              
                try:
                    img_pil = Image.open(io.BytesIO(image_bytes))
                    
                    # Convert to grayscale for better OCR
                    img_pil = img_pil.convert('L')
                    
                    ocr_text = pytesseract.image_to_string(img_pil)
                    
                    if ocr_text.strip():
                        images_data.append({
                            'type': 'image',
                            'content': ocr_text,
                            'page': page_num + 1,
                            'image_path': image_filename,
                            'source': f'Image on Page {page_num + 1}'
                        })
                except Exception as e:
                    print(f"OCR failed on page {page_num + 1}: {e}")
        
        return images_data
    
    def process_document(self):
        """
        Process entire document with all modalities.
        """
        print(f"Processing document: {self.pdf_path}")
        
        text_chunks = self.extract_text_chunks()
        print(f"Extracted {len(text_chunks)} text chunks")
        
        tables = self.extract_tables()
        print(f"Extracted {len(tables)} tables")
        
        images = self.extract_images_with_ocr()
        print(f"Extracted {len(images)} images with OCR")
        
        all_chunks = text_chunks + tables + images
        
        # Sort by page number
        all_chunks.sort(key=lambda x: x['page'])
        
        print(f"Total chunks: {len(all_chunks)}")
        
        return all_chunks
    
    def close(self):
        self.doc.close()

if __name__ == "__main__":
    processor = DocumentProcessor("data/raw/qatar_test_doc.pdf")
    chunks = processor.process_document()
    print(f"\nSample chunk: {chunks[0]}")
    processor.close()