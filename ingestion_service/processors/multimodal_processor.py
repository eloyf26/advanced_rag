"""
Multi-Modal File Processor Implementation
File: ingestion_service/processors/multimodal_processor.py
"""

import asyncio
import logging
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import json

import aiofiles
from llama_index.core import Document
from llama_index.readers.file import (
    PDFReader, DocxReader, UnstructuredReader, CSVReader, 
    HTMLTagReader, MarkdownReader, JSONReader
)
import pandas as pd
from PIL import Image
import whisper

from config import IngestionConfig

logger = logging.getLogger(__name__)


class MultiModalFileProcessor:
    """
    Processes multiple file types with specialized handlers for each format
    """
    
    def __init__(self, config: IngestionConfig):
        self.config = config
        
        # Initialize readers
        self.pdf_reader = PDFReader()
        self.docx_reader = DocxReader()
        self.unstructured_reader = UnstructuredReader()
        self.csv_reader = CSVReader()
        self.html_reader = HTMLTagReader()
        self.markdown_reader = MarkdownReader()
        self.json_reader = JSONReader()
        
        # Initialize OCR and speech recognition (if enabled)
        self.ocr_model = None
        self.whisper_model = None
        
        if config.enable_ocr:
            try:
                import easyocr
                self.ocr_model = easyocr.Reader(['en'])
                logger.info("OCR model initialized")
            except ImportError:
                logger.warning("EasyOCR not available, image processing disabled")
        
        if config.enable_speech_to_text:
            try:
                self.whisper_model = whisper.load_model("base")
                logger.info("Whisper model initialized")
            except Exception as e:
                logger.warning(f"Whisper not available: {e}")
        
        # File type processors mapping
        self.processors = {
            # Documents
            'pdf': self._process_pdf,
            'docx': self._process_docx,
            'doc': self._process_doc,
            'txt': self._process_text,
            'md': self._process_markdown,
            'rtf': self._process_rtf,
            
            # Spreadsheets
            'csv': self._process_csv,
            'xlsx': self._process_excel,
            'xls': self._process_excel,
            
            # Images
            'jpg': self._process_image,
            'jpeg': self._process_image,
            'png': self._process_image,
            'gif': self._process_image,
            'bmp': self._process_image,
            'tiff': self._process_image,
            
            # Audio
            'mp3': self._process_audio,
            'wav': self._process_audio,
            'm4a': self._process_audio,
            'flac': self._process_audio,
            
            # Code
            'py': self._process_code,
            'js': self._process_code,
            'java': self._process_code,
            'cpp': self._process_code,
            'c': self._process_code,
            'cs': self._process_code,
            'php': self._process_code,
            'rb': self._process_code,
            'go': self._process_code,
            'sql': self._process_code,
            
            # Structured data
            'json': self._process_json,
            'xml': self._process_xml,
            'yaml': self._process_yaml,
            'yml': self._process_yaml,
            
            # Web
            'html': self._process_html,
            'htm': self._process_html,
            'css': self._process_css,
            
            # Archives
            'zip': self._process_archive,
            'tar': self._process_archive,
            'gz': self._process_archive,
        }
        
        logger.info(f"Multi-modal processor initialized with {len(self.processors)} file type handlers")
    
    async def process_file(self, file_path: str) -> List[Document]:
        """
        Main file processing method that dispatches to appropriate processor
        """
        try:
            file_path_obj = Path(file_path)
            
            # Validate file
            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if file_path_obj.stat().st_size > self.config.max_file_size_mb * 1024 * 1024:
                raise ValueError(f"File too large: {file_path}")
            
            # Determine file type
            file_extension = file_path_obj.suffix.lower().lstrip('.')
            mime_type = mimetypes.guess_type(file_path)[0]
            
            logger.debug(f"Processing {file_path}: type={file_extension}, mime={mime_type}")
            
            # Get appropriate processor
            processor = self.processors.get(file_extension, self._process_unknown)
            
            # Process file
            documents = await processor(file_path, file_extension, mime_type)
            
            # Add common metadata
            for doc in documents:
                doc.metadata.update({
                    'file_path': str(file_path),
                    'file_name': file_path_obj.name,
                    'file_type': file_extension,
                    'file_size': file_path_obj.stat().st_size,
                    'mime_type': mime_type,
                    'extraction_method': processor.__name__
                })
            
            logger.debug(f"Extracted {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []
    
    # Document processors
    async def _process_pdf(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process PDF files"""
        try:
            documents = await asyncio.to_thread(self.pdf_reader.load_data, file_path)
            return documents
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            # Fallback to unstructured reader
            return await self._process_with_unstructured(file_path)
    
    async def _process_docx(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process DOCX files"""
        try:
            documents = await asyncio.to_thread(self.docx_reader.load_data, file_path)
            return documents
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {e}")
            return await self._process_with_unstructured(file_path)
    
    async def _process_doc(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process DOC files"""
        return await self._process_with_unstructured(file_path)
    
    async def _process_text(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process plain text files"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            return [Document(text=content, metadata={'extraction_method': 'direct_text'})]
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                        content = await f.read()
                    return [Document(text=content, metadata={'extraction_method': f'text_{encoding}'})]
                except:
                    continue
            
            logger.error(f"Could not decode text file {file_path}")
            return []
    
    async def _process_markdown(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process Markdown files"""
        try:
            documents = await asyncio.to_thread(self.markdown_reader.load_data, file_path)
            return documents
        except Exception as e:
            logger.error(f"Error processing Markdown {file_path}: {e}")
            return await self._process_text(file_path, file_type, mime_type)
    
    async def _process_rtf(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process RTF files"""
        return await self._process_with_unstructured(file_path)
    
    # Spreadsheet processors
    async def _process_csv(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process CSV files"""
        try:
            # Use pandas for better CSV handling
            df = await asyncio.to_thread(pd.read_csv, file_path)
            
            documents = []
            
            # Create summary document
            summary = f"CSV file with {len(df)} rows and {len(df.columns)} columns.\n"
            summary += f"Columns: {', '.join(df.columns)}\n\n"
            
            # Add sample data
            if len(df) > 0:
                summary += "Sample data:\n"
                summary += df.head(5).to_string(index=False)
            
            documents.append(Document(
                text=summary,
                metadata={
                    'extraction_method': 'csv_summary',
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'columns': list(df.columns)
                }
            ))
            
            # If small enough, include full data as text
            if len(df) <= 1000:  # Configurable threshold
                full_text = df.to_string(index=False)
                documents.append(Document(
                    text=full_text,
                    metadata={
                        'extraction_method': 'csv_full',
                        'row_count': len(df),
                        'column_count': len(df.columns)
                    }
                ))
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing CSV {file_path}: {e}")
            # Fallback to simple text processing
            return await self._process_text(file_path, file_type, mime_type)
    
    async def _process_excel(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process Excel files"""
        try:
            # Read all sheets
            excel_data = await asyncio.to_thread(pd.read_excel, file_path, sheet_name=None)
            
            documents = []
            
            for sheet_name, df in excel_data.items():
                # Create document for each sheet
                summary = f"Excel sheet '{sheet_name}' with {len(df)} rows and {len(df.columns)} columns.\n"
                summary += f"Columns: {', '.join(df.columns)}\n\n"
                
                if len(df) > 0:
                    summary += "Sample data:\n"
                    summary += df.head(5).to_string(index=False)
                
                documents.append(Document(
                    text=summary,
                    metadata={
                        'extraction_method': 'excel_sheet',
                        'sheet_name': sheet_name,
                        'row_count': len(df),
                        'column_count': len(df.columns),
                        'columns': list(df.columns)
                    }
                ))
                
                # Include full data for smaller sheets
                if len(df) <= 500:
                    full_text = df.to_string(index=False)
                    documents.append(Document(
                        text=full_text,
                        metadata={
                            'extraction_method': 'excel_full',
                            'sheet_name': sheet_name,
                            'row_count': len(df),
                            'column_count': len(df.columns)
                        }
                    ))
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing Excel {file_path}: {e}")
            return []
    
    # Image processors
    async def _process_image(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process image files with OCR"""
        if not self.ocr_model:
            logger.warning(f"OCR not available, skipping image {file_path}")
            return []
        
        try:
            # Perform OCR
            results = await asyncio.to_thread(self.ocr_model.readtext, file_path)
            
            # Extract text with confidence scores
            extracted_text = []
            high_confidence_text = []
            
            for (bbox, text, confidence) in results:
                extracted_text.append(f"{text} (confidence: {confidence:.2f})")
                if confidence >= self.config.ocr_confidence_threshold:
                    high_confidence_text.append(text)
            
            if not extracted_text:
                logger.warning(f"No text extracted from image {file_path}")
                return []
            
            # Create documents
            documents = []
            
            # Full OCR results
            full_text = "\n".join(extracted_text)
            documents.append(Document(
                text=full_text,
                metadata={
                    'extraction_method': 'ocr_full',
                    'ocr_confidence_threshold': self.config.ocr_confidence_threshold,
                    'total_text_regions': len(results)
                }
            ))
            
            # High confidence text only
            if high_confidence_text:
                clean_text = "\n".join(high_confidence_text)
                documents.append(Document(
                    text=clean_text,
                    metadata={
                        'extraction_method': 'ocr_high_confidence',
                        'confidence_threshold': self.config.ocr_confidence_threshold,
                        'high_confidence_regions': len(high_confidence_text)
                    }
                ))
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {e}")
            return []
    
    # Audio processors
    async def _process_audio(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process audio files with speech-to-text"""
        if not self.whisper_model:
            logger.warning(f"Whisper not available, skipping audio {file_path}")
            return []
        
        try:
            # Transcribe audio
            result = await asyncio.to_thread(self.whisper_model.transcribe, file_path)
            
            # Extract transcription
            full_text = result["text"]
            
            if not full_text.strip():
                logger.warning(f"No speech detected in audio {file_path}")
                return []
            
            documents = []
            
            # Full transcription
            documents.append(Document(
                text=full_text,
                metadata={
                    'extraction_method': 'whisper_transcription',
                    'language': result.get("language", "unknown"),
                    'duration': result.get("duration", 0)
                }
            ))
            
            # Segment-based transcription if available
            if "segments" in result:
                segments_text = []
                for segment in result["segments"]:
                    start_time = segment.get("start", 0)
                    end_time = segment.get("end", 0)
                    text = segment.get("text", "")
                    segments_text.append(f"[{start_time:.1f}s - {end_time:.1f}s]: {text}")
                
                if segments_text:
                    timestamped_text = "\n".join(segments_text)
                    documents.append(Document(
                        text=timestamped_text,
                        metadata={
                            'extraction_method': 'whisper_timestamped',
                            'segment_count': len(result["segments"]),
                            'total_duration': result.get("duration", 0)
                        }
                    ))
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing audio {file_path}: {e}")
            return []
    
    # Code processors
    async def _process_code(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process code files"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Basic code analysis
            lines = content.split('\n')
            
            # Extract functions/classes (simple regex-based)
            functions = []
            classes = []
            imports = []
            
            for line_num, line in enumerate(lines, 1):
                line_stripped = line.strip()
                
                # Python-style analysis (can be extended for other languages)
                if line_stripped.startswith('def '):
                    functions.append(f"Line {line_num}: {line_stripped}")
                elif line_stripped.startswith('class '):
                    classes.append(f"Line {line_num}: {line_stripped}")
                elif line_stripped.startswith(('import ', 'from ')):
                    imports.append(f"Line {line_num}: {line_stripped}")
            
            # Create summary
            summary = f"Code file ({file_type}) with {len(lines)} lines.\n"
            if imports:
                summary += f"\nImports ({len(imports)}):\n" + "\n".join(imports[:10])
            if classes:
                summary += f"\nClasses ({len(classes)}):\n" + "\n".join(classes[:10])
            if functions:
                summary += f"\nFunctions ({len(functions)}):\n" + "\n".join(functions[:10])
            
            documents = [
                Document(
                    text=content,
                    metadata={
                        'extraction_method': 'code_full',
                        'language': file_type,
                        'line_count': len(lines),
                        'function_count': len(functions),
                        'class_count': len(classes),
                        'import_count': len(imports)
                    }
                ),
                Document(
                    text=summary,
                    metadata={
                        'extraction_method': 'code_summary',
                        'language': file_type,
                        'analysis_type': 'structure'
                    }
                )
            ]
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing code file {file_path}: {e}")
            return await self._process_text(file_path, file_type, mime_type)
    
    # Structured data processors
    async def _process_json(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process JSON files"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            data = json.loads(content)
            
            documents = []
            
            # Full JSON as text
            formatted_json = json.dumps(data, indent=2, ensure_ascii=False)
            documents.append(Document(
                text=formatted_json,
                metadata={
                    'extraction_method': 'json_formatted',
                    'data_type': type(data).__name__
                }
            ))
            
            # JSON summary
            summary = self._analyze_json_structure(data)
            documents.append(Document(
                text=summary,
                metadata={
                    'extraction_method': 'json_summary',
                    'data_type': type(data).__name__
                }
            ))
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing JSON {file_path}: {e}")
            return await self._process_text(file_path, file_type, mime_type)
    
    async def _process_xml(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process XML files"""
        try:
            import xml.etree.ElementTree as ET
            
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Parse XML
            root = ET.fromstring(content)
            
            # Extract text content
            text_content = []
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    text_content.append(elem.text.strip())
            
            # Create summary
            summary = f"XML document with root element '{root.tag}'\n"
            summary += f"Total elements: {len(list(root.iter()))}\n"
            summary += f"Text content:\n" + "\n".join(text_content[:20])
            
            documents = [
                Document(
                    text=content,
                    metadata={
                        'extraction_method': 'xml_raw',
                        'root_element': root.tag,
                        'element_count': len(list(root.iter()))
                    }
                ),
                Document(
                    text=summary,
                    metadata={
                        'extraction_method': 'xml_summary',
                        'root_element': root.tag
                    }
                )
            ]
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing XML {file_path}: {e}")
            return await self._process_text(file_path, file_type, mime_type)
    
    async def _process_yaml(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process YAML files"""
        try:
            import yaml
            
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            data = yaml.safe_load(content)
            
            # Create formatted representation
            formatted_yaml = yaml.dump(data, default_flow_style=False, allow_unicode=True)
            
            # Create summary
            summary = f"YAML document\n"
            summary += f"Data type: {type(data).__name__}\n"
            if isinstance(data, dict):
                summary += f"Keys: {', '.join(str(k) for k in list(data.keys())[:10])}\n"
            
            documents = [
                Document(
                    text=formatted_yaml,
                    metadata={
                        'extraction_method': 'yaml_formatted',
                        'data_type': type(data).__name__
                    }
                ),
                Document(
                    text=summary,
                    metadata={
                        'extraction_method': 'yaml_summary',
                        'data_type': type(data).__name__
                    }
                )
            ]
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing YAML {file_path}: {e}")
            return await self._process_text(file_path, file_type, mime_type)
    
    # Web content processors
    async def _process_html(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process HTML files"""
        try:
            documents = await asyncio.to_thread(self.html_reader.load_data, file_path)
            return documents
        except Exception as e:
            logger.error(f"Error processing HTML {file_path}: {e}")
            return await self._process_text(file_path, file_type, mime_type)
    
    async def _process_css(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process CSS files"""
        return await self._process_text(file_path, file_type, mime_type)
    
    # Archive processors
    async def _process_archive(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process archive files by extracting and processing contents"""
        try:
            import zipfile
            import tarfile
            
            documents = []
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract archive
                if file_type == 'zip':
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                elif file_type in ['tar', 'gz']:
                    with tarfile.open(file_path, 'r:*') as tar_ref:
                        tar_ref.extractall(temp_dir)
                
                # Process extracted files
                temp_path = Path(temp_dir)
                for extracted_file in temp_path.rglob('*'):
                    if extracted_file.is_file() and extracted_file.stat().st_size < self.config.max_file_size_mb * 1024 * 1024:
                        try:
                            file_docs = await self.process_file(str(extracted_file))
                            for doc in file_docs:
                                # Add archive context to metadata
                                doc.metadata.update({
                                    'archive_source': file_path,
                                    'extracted_from': str(extracted_file.relative_to(temp_path))
                                })
                                documents.append(doc)
                        except Exception as e:
                            logger.warning(f"Error processing extracted file {extracted_file}: {e}")
                            continue
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing archive {file_path}: {e}")
            return []
    
    # Fallback processors
    async def _process_with_unstructured(self, file_path: str) -> List[Document]:
        """Process file with unstructured reader as fallback"""
        try:
            documents = await asyncio.to_thread(self.unstructured_reader.load_data, file_path)
            return documents
        except Exception as e:
            logger.error(f"Unstructured reader failed for {file_path}: {e}")
            return []
    
    async def _process_unknown(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process unknown file types"""
        logger.warning(f"Unknown file type: {file_type} for {file_path}")
        
        # Try to process as text first
        try:
            return await self._process_text(file_path, file_type, mime_type)
        except:
            # Try unstructured reader
            return await self._process_with_unstructured(file_path)
    
    def _analyze_json_structure(self, data: Any, max_depth: int = 3, current_depth: int = 0) -> str:
        """Analyze JSON structure and create summary"""
        if current_depth >= max_depth:
            return f"... (max depth {max_depth} reached)"
        
        if isinstance(data, dict):
            summary = f"Object with {len(data)} keys: {', '.join(list(data.keys())[:10])}\n"
            for key, value in list(data.items())[:5]:  # Limit to first 5 items
                summary += f"  {key}: {type(value).__name__}"
                if isinstance(value, (dict, list)) and current_depth < max_depth - 1:
                    summary += f" - {self._analyze_json_structure(value, max_depth, current_depth + 1)}"
                summary += "\n"
        elif isinstance(data, list):
            summary = f"Array with {len(data)} items"
            if data:
                summary += f", first item type: {type(data[0]).__name__}"
        else:
            summary = f"{type(data).__name__}: {str(data)[:100]}"
        
        return summary
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported file types"""
        return list(self.processors.keys())
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            'supported_types': len(self.processors),
            'ocr_enabled': self.ocr_model is not None,
            'speech_to_text_enabled': self.whisper_model is not None,
            'max_file_size_mb': self.config.max_file_size_mb,
            'ocr_confidence_threshold': self.config.ocr_confidence_threshold,
            'processors': list(self.processors.keys())
        }