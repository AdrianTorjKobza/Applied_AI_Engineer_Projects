# PDF Parser

import os
import ollama
from PIL import Image
from docling.document_converter import DocumentConverter, PdfFormatOption # Docling is a powerfull library that converts complex PDFs into structured data.
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import PictureItem

class MultimodalParser:
    def __init__(self, output_dir="storage/images"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True) # Creates the folder if it doesn't exist.
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False # Skip reading text inside images (faster).
        pipeline_options.do_table_structure = False # Skip complex table formatting.
        
        # Extract the images and keep them at original size.
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 1.0  
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def process_pdf(self, pdf_path):
        try:
            result = self.converter.convert(pdf_path) # Convert PDF to a digital object.
            doc = result.document
            
            text_chunks = []
            image_metadata = []

            # Semantic Text Chunking (Markdown Export).
            content_md = doc.export_to_markdown() # Turn PDF into markdown text.
            chunk_size = 1200
            overlap = 200

            for i in range(0, len(content_md), chunk_size - overlap):
                chunk = content_md[i : i + chunk_size]
                text_chunks.append({
                    "id": f"text_{i}_{os.path.basename(pdf_path)}",
                    "content": chunk,
                    "metadata": {"source": pdf_path}
                })

            image_counter = 0

            for item, level in doc.iterate_items(traverse_pictures=True):
                # Check if the item is a PictureItem and has an image rendered.
                if isinstance(item, PictureItem):
                    # We get the image using the specialized get_image method.
                    rendered_image = item.get_image(doc)
                    
                    if rendered_image:
                        image_counter += 1
                        img_name = f"fig_{image_counter}_{os.path.basename(pdf_path)}.png"
                        img_path = os.path.join(self.output_dir, img_name)
                        
                        # Save the PIL image object
                        rendered_image.save(img_path)
                        
                        # Use Ollama Moondream to caption the figure.
                        # Moondream is a tiny but "smart" vision model.
                        try:
                            response = ollama.generate(
                                model='moondream',
                                prompt="This is a diagram from a pdf manual or book. Describe it for a search system.",
                                images=[img_path]
                            )
                            caption = response['response']
                        except Exception:
                            caption = "Image from documentation."
                        
                        image_metadata.append({
                            "id": f"img_{image_counter}_{os.path.basename(pdf_path)}",
                            "caption": caption,
                            "path": img_path,
                            "metadata": {"type": "diagram"}
                        })
            
            return text_chunks, image_metadata
            
        except Exception as e:
            print(f"Parsing failed: {str(e)}")
            return [], []