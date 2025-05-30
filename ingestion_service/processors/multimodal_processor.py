"""
Extended Multi-Modal File Processor Implementation
File: ingestion_service/processors/multimodal_processor.py
Supports many file types with specialized handlers
"""

import asyncio
import logging
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import json
import base64

import aiofiles
from llama_index.core import Document
from llama_index.readers.file import (
    PDFReader, DocxReader, UnstructuredReader, CSVReader, 
    HTMLTagReader, MarkdownReader, HWPReader, EpubReader,
    FlatReader, ImageCaptionReader, ImageReader, ImageVisionLLMReader,
    IPYNBReader, MboxReader, PptxReader, PandasCSVReader,
    VideoAudioReader, PyMuPDFReader, ImageTabularChartReader,
    XMLReader, PagedCSVReader, RTFReader
)
import pandas as pd
from PIL import Image
import whisper

from config import IngestionConfig

logger = logging.getLogger(__name__)


class MultiModalFileProcessor:
    """
    Processes 30+ file types with specialized handlers for each format
    """
    
    def __init__(self, config: IngestionConfig):
        self.config = config
        
        # Initialize all available readers
        self.pdf_reader = PDFReader()
        self.pymupdf_reader = PyMuPDFReader()  # Alternative PDF reader
        self.docx_reader = DocxReader()
        self.hwp_reader = HWPReader()  # Korean Word Processor
        self.pptx_reader = PptxReader()  # PowerPoint
        self.rtf_reader = RTFReader()  # Rich Text Format
        
        # Spreadsheet readers
        self.csv_reader = CSVReader()
        self.pandas_csv_reader = PandasCSVReader()
        self.paged_csv_reader = PagedCSVReader()  # For large CSVs
        
        # Web/Markup readers
        self.html_reader = HTMLTagReader()
        self.markdown_reader = MarkdownReader()
        self.xml_reader = XMLReader()
        
        # eBook readers
        self.epub_reader = EpubReader()
        
        # Notebook readers
        self.ipynb_reader = IPYNBReader()
        
        # Email readers
        self.mbox_reader = MboxReader()
        
        # Image readers with different capabilities
        self.image_reader = ImageReader()
        self.image_caption_reader = ImageCaptionReader()
        self.image_vision_llm_reader = ImageVisionLLMReader()
        self.image_tabular_chart_reader = ImageTabularChartReader()
        
        # Video/Audio readers
        self.video_audio_reader = VideoAudioReader()
        
        # Generic readers
        self.flat_reader = FlatReader()
        self.unstructured_reader = UnstructuredReader()
        
        # Initialize OCR and speech recognition (if enabled)
        self.ocr_model = None
        self.whisper_model = None
        
        if config.enable_ocr:
            try:
                import easyocr
                self.ocr_model = easyocr.Reader(['en'])
                logger.info("OCR model initialized")
            except ImportError:
                logger.warning("EasyOCR not available, some image processing features disabled")
        
        if config.enable_speech_to_text:
            try:
                self.whisper_model = whisper.load_model("base")
                logger.info("Whisper model initialized")
            except Exception as e:
                logger.warning(f"Whisper not available: {e}")
        
        # Enhanced file type processors mapping
        self.processors = {
            # Documents
            'pdf': self._process_pdf,
            'docx': self._process_docx,
            'doc': self._process_doc,
            'hwp': self._process_hwp,  # Korean Word Processor
            'pptx': self._process_pptx,  # PowerPoint
            'ppt': self._process_ppt,    # Legacy PowerPoint
            'txt': self._process_text,
            'md': self._process_markdown,
            'rtf': self._process_rtf,
            'odt': self._process_odt,    # OpenDocument Text
            'tex': self._process_latex,   # LaTeX
            
            # Spreadsheets
            'csv': self._process_csv,
            'xlsx': self._process_excel,
            'xls': self._process_excel,
            'ods': self._process_ods,    # OpenDocument Spreadsheet
            'tsv': self._process_tsv,    # Tab-separated values
            
            # eBooks
            'epub': self._process_epub,
            'mobi': self._process_mobi,  # Kindle format
            'azw': self._process_azw,    # Amazon format
            'azw3': self._process_azw3,  # Amazon format
            'fb2': self._process_fb2,    # FictionBook
            
            # Images
            'jpg': self._process_image,
            'jpeg': self._process_image,
            'png': self._process_image,
            'gif': self._process_image,
            'bmp': self._process_image,
            'tiff': self._process_image,
            'tif': self._process_image,
            'webp': self._process_image,
            'svg': self._process_svg,
            'ico': self._process_image,
            
            # Audio
            'mp3': self._process_audio,
            'wav': self._process_audio,
            'm4a': self._process_audio,
            'flac': self._process_audio,
            'ogg': self._process_audio,
            'wma': self._process_audio,
            'aac': self._process_audio,
            'opus': self._process_audio,
            
            # Video
            'mp4': self._process_video,
            'avi': self._process_video,
            'mkv': self._process_video,
            'mov': self._process_video,
            'wmv': self._process_video,
            'flv': self._process_video,
            'webm': self._process_video,
            'm4v': self._process_video,
            
            # Code/Notebooks
            'py': self._process_code,
            'ipynb': self._process_ipynb,  # Jupyter notebooks
            'js': self._process_code,
            'java': self._process_code,
            'cpp': self._process_code,
            'c': self._process_code,
            'cs': self._process_code,
            'php': self._process_code,
            'rb': self._process_code,
            'go': self._process_code,
            'rs': self._process_code,
            'swift': self._process_code,
            'kt': self._process_code,
            'scala': self._process_code,
            'r': self._process_code,
            'sql': self._process_code,
            'sh': self._process_code,
            'bash': self._process_code,
            'ps1': self._process_code,
            
            # Structured data
            'json': self._process_json,
            'xml': self._process_xml_structured,
            'yaml': self._process_yaml,
            'yml': self._process_yaml,
            'toml': self._process_toml,
            'ini': self._process_ini,
            'cfg': self._process_config,
            'conf': self._process_config,
            
            # Web
            'html': self._process_html,
            'htm': self._process_html,
            'css': self._process_css,
            'scss': self._process_scss,
            'less': self._process_less,
            'xhtml': self._process_xhtml,
            
            # Email
            'mbox': self._process_mbox,
            'eml': self._process_eml,
            'msg': self._process_msg,
            
            # Archives
            'zip': self._process_archive,
            'tar': self._process_archive,
            'gz': self._process_archive,
            'bz2': self._process_archive,
            'xz': self._process_archive,
            'rar': self._process_archive,
            '7z': self._process_archive,
            
            # Other formats
            'log': self._process_log,
            'pcap': self._process_pcap,  # Network capture
            'vcf': self._process_vcf,    # vCard
            'ics': self._process_ics,    # iCalendar
        }
        
        logger.info(f"Extended multi-modal processor initialized with {len(self.processors)} file type handlers")
    
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
        """Process PDF files with fallback to PyMuPDF"""
        try:
            documents = await asyncio.to_thread(self.pdf_reader.load_data, file_path)
            return documents
        except Exception as e:
            logger.warning(f"PDFReader failed, trying PyMuPDFReader: {e}")
            try:
                documents = await asyncio.to_thread(self.pymupdf_reader.load_data, file_path)
                return documents
            except Exception as e2:
                logger.error(f"Both PDF readers failed: {e2}")
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
        try:
            documents = await asyncio.to_thread(self.rtf_reader.load_data, file_path)
            return documents
        except Exception as e:
            logger.error(f"Error processing RTF {file_path}: {e}")
            return await self._process_with_unstructured(file_path)
    
    async def _process_hwp(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process HWP (Korean Word Processor) files"""
        try:
            documents = await asyncio.to_thread(self.hwp_reader.load_data, file_path)
            return documents
        except Exception as e:
            logger.error(f"Error processing HWP {file_path}: {e}")
            return await self._process_with_unstructured(file_path)
    
    async def _process_pptx(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process PowerPoint files"""
        try:
            documents = await asyncio.to_thread(self.pptx_reader.load_data, file_path)
            return documents
        except Exception as e:
            logger.error(f"Error processing PPTX {file_path}: {e}")
            return await self._process_with_unstructured(file_path)
    
    async def _process_ppt(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process legacy PowerPoint files"""
        return await self._process_with_unstructured(file_path)
    
    async def _process_odt(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process OpenDocument Text files"""
        return await self._process_with_unstructured(file_path)
    
    async def _process_latex(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process LaTeX files"""
        return await self._process_text(file_path, file_type, mime_type)
    
    # Spreadsheet processors
    async def _process_csv(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process CSV files with multiple reader options"""
        try:
            # Check file size to decide which reader to use
            file_size = Path(file_path).stat().st_size
            
            if file_size > 50 * 1024 * 1024:  # 50MB threshold for paged reader
                logger.info(f"Using PagedCSVReader for large file ({file_size} bytes)")
                documents = await asyncio.to_thread(self.paged_csv_reader.load_data, file_path)
                return documents
            else:
                # Use PandasCSVReader for better handling
                documents = await asyncio.to_thread(self.pandas_csv_reader.load_data, file_path)
                return documents
                
        except Exception as e:
            logger.error(f"Error with CSV readers, falling back to manual processing: {e}")
            return await self._process_csv_manual(file_path, file_type, mime_type)
    
    async def _process_csv_manual(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Manual CSV processing as fallback"""
        try:
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
            if len(df) <= 1000:
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
            logger.error(f"Error in manual CSV processing {file_path}: {e}")
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
                summary += f"Columns: {', '.join(str(c) for c in df.columns)}\n\n"
                
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
    
    async def _process_tsv(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process TSV files"""
        try:
            df = await asyncio.to_thread(pd.read_csv, file_path, sep='\t')
            # Similar processing as CSV
            return await self._process_csv_manual(file_path, file_type, mime_type)
        except Exception as e:
            logger.error(f"Error processing TSV {file_path}: {e}")
            return await self._process_text(file_path, file_type, mime_type)
    
    async def _process_ods(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process OpenDocument Spreadsheet files"""
        try:
            df = await asyncio.to_thread(pd.read_excel, file_path, engine='odf')
            # Process similar to Excel
            return await self._process_excel(file_path, file_type, mime_type)
        except Exception as e:
            logger.error(f"Error processing ODS {file_path}: {e}")
            return await self._process_with_unstructured(file_path)
    
    # Audio processors
    async def _process_audio(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process audio files with speech-to-text"""
        # First try VideoAudioReader for better audio handling
        try:
            documents = await asyncio.to_thread(self.video_audio_reader.load_data, file_path)
            if documents:
                return documents
        except Exception as e:
            logger.warning(f"VideoAudioReader failed for audio, trying Whisper: {e}")
        
        # Fallback to Whisper
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
    
    # Video processors
    async def _process_video(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process video files with audio extraction"""
        try:
            documents = await asyncio.to_thread(self.video_audio_reader.load_data, file_path)
            return documents
        except Exception as e:
            logger.error(f"Error processing video {file_path}: {e}")
            
            # Fallback to audio extraction only
            if self.whisper_model:
                return await self._process_audio(file_path, file_type, mime_type)
            return []
    
    # Image processors
    async def _process_image(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process image files with multiple strategies"""
        documents = []
        
        # Strategy 1: Basic image reading
        try:
            basic_docs = await asyncio.to_thread(self.image_reader.load_data, file_path)
            documents.extend(basic_docs)
        except Exception as e:
            logger.warning(f"Basic image reading failed: {e}")
        
        # Strategy 2: Image captioning
        if self.config.enable_image_captioning:
            try:
                caption_docs = await asyncio.to_thread(self.image_caption_reader.load_data, file_path)
                documents.extend(caption_docs)
            except Exception as e:
                logger.warning(f"Image captioning failed: {e}")
        
        # Strategy 3: Vision LLM analysis
        if self.config.enable_vision_llm:
            try:
                vision_docs = await asyncio.to_thread(self.image_vision_llm_reader.load_data, file_path)
                documents.extend(vision_docs)
            except Exception as e:
                logger.warning(f"Vision LLM analysis failed: {e}")
        
        # Strategy 4: Tabular/Chart extraction
        try:
            chart_docs = await asyncio.to_thread(self.image_tabular_chart_reader.load_data, file_path)
            documents.extend(chart_docs)
        except Exception as e:
            logger.warning(f"Chart/table extraction failed: {e}")
        
        # Strategy 5: OCR (existing implementation)
        if self.ocr_model:
            try:
                ocr_docs = await self._process_image_ocr(file_path, file_type, mime_type)
                documents.extend(ocr_docs)
            except Exception as e:
                logger.warning(f"OCR processing failed: {e}")
        
        return documents
    
    async def _process_image_ocr(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """OCR processing for images"""
        if not self.ocr_model:
            return []
        
        try:
            results = await asyncio.to_thread(self.ocr_model.readtext, file_path)
            
            extracted_text = []
            high_confidence_text = []
            
            for (bbox, text, confidence) in results:
                extracted_text.append(f"{text} (confidence: {confidence:.2f})")
                if confidence >= self.config.ocr_confidence_threshold:
                    high_confidence_text.append(text)
            
            if not extracted_text:
                return []
            
            documents = []
            
            full_text = "\n".join(extracted_text)
            documents.append(Document(
                text=full_text,
                metadata={
                    'extraction_method': 'ocr_full',
                    'ocr_confidence_threshold': self.config.ocr_confidence_threshold,
                    'total_text_regions': len(results)
                }
            ))
            
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
            logger.error(f"Error in OCR processing {file_path}: {e}")
            return []
    
    async def _process_svg(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process SVG files"""
        return await self._process_xml_structured(file_path, file_type, mime_type)
    
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
    
    # Notebook processors
    async def _process_ipynb(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process Jupyter notebooks"""
        try:
            documents = await asyncio.to_thread(self.ipynb_reader.load_data, file_path)
            return documents
        except Exception as e:
            logger.error(f"Error processing notebook {file_path}: {e}")
            return await self._process_json(file_path, file_type, mime_type)
    
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
    
    async def _process_xml(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process XML files (manual fallback)"""
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
    
    async def _process_xml_structured(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process XML files with XMLReader"""
        try:
            documents = await asyncio.to_thread(self.xml_reader.load_data, file_path)
            return documents
        except Exception as e:
            logger.error(f"Error with XMLReader, falling back to manual: {e}")
            return await self._process_xml(file_path, file_type, mime_type)
    
    async def _process_toml(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process TOML files"""
        try:
            import toml
            
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            data = toml.loads(content)
            
            formatted_toml = toml.dumps(data)
            
            documents = [
                Document(
                    text=formatted_toml,
                    metadata={
                        'extraction_method': 'toml_formatted',
                        'data_type': type(data).__name__
                    }
                ),
                Document(
                    text=self._analyze_json_structure(data),
                    metadata={
                        'extraction_method': 'toml_summary',
                        'data_type': type(data).__name__
                    }
                )
            ]
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing TOML {file_path}: {e}")
            return await self._process_text(file_path, file_type, mime_type)
    
    async def _process_ini(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process INI configuration files"""
        try:
            import configparser
            
            config = configparser.ConfigParser()
            
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            config.read_string(content)
            
            # Create structured representation
            ini_data = {}
            for section in config.sections():
                ini_data[section] = dict(config.items(section))
            
            formatted_ini = json.dumps(ini_data, indent=2)
            
            documents = [
                Document(
                    text=content,
                    metadata={
                        'extraction_method': 'ini_raw',
                        'section_count': len(config.sections())
                    }
                ),
                Document(
                    text=formatted_ini,
                    metadata={
                        'extraction_method': 'ini_structured',
                        'sections': config.sections()
                    }
                )
            ]
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing INI {file_path}: {e}")
            return await self._process_text(file_path, file_type, mime_type)
    
    async def _process_config(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process generic config files"""
        # Try INI parsing first
        docs = await self._process_ini(file_path, file_type, mime_type)
        if docs:
            return docs
        # Fallback to text
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
    
    async def _process_xhtml(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process XHTML files"""
        return await self._process_html(file_path, file_type, mime_type)
    
    async def _process_scss(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process SCSS files"""
        return await self._process_code(file_path, file_type, mime_type)
    
    async def _process_less(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process LESS files"""
        return await self._process_code(file_path, file_type, mime_type)
    
    # eBook processors
    async def _process_epub(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process EPUB files"""
        try:
            documents = await asyncio.to_thread(self.epub_reader.load_data, file_path)
            return documents
        except Exception as e:
            logger.error(f"Error processing EPUB {file_path}: {e}")
            return await self._process_with_unstructured(file_path)
    
    async def _process_mobi(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process MOBI files"""
        return await self._process_with_unstructured(file_path)
    
    async def _process_azw(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process AZW files"""
        return await self._process_with_unstructured(file_path)
    
    async def _process_azw3(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process AZW3 files"""
        return await self._process_with_unstructured(file_path)
    
    async def _process_fb2(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process FictionBook files"""
        return await self._process_xml_structured(file_path, file_type, mime_type)
    
    # Email processors
    async def _process_mbox(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process MBOX email archives"""
        try:
            documents = await asyncio.to_thread(self.mbox_reader.load_data, file_path)
            return documents
        except Exception as e:
            logger.error(f"Error processing MBOX {file_path}: {e}")
            return []
    
    async def _process_eml(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process EML email files"""
        try:
            import email
            
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            msg = email.message_from_string(content)
            
            # Extract email content
            subject = msg.get('Subject', 'No Subject')
            from_addr = msg.get('From', 'Unknown')
            to_addr = msg.get('To', 'Unknown')
            date = msg.get('Date', 'Unknown')
            
            # Extract body
            body = ""
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
            
            email_text = f"Subject: {subject}\nFrom: {from_addr}\nTo: {to_addr}\nDate: {date}\n\n{body}"
            
            return [Document(
                text=email_text,
                metadata={
                    'extraction_method': 'eml_parser',
                    'subject': subject,
                    'from': from_addr,
                    'to': to_addr,
                    'date': date
                }
            )]
            
        except Exception as e:
            logger.error(f"Error processing EML {file_path}: {e}")
            return await self._process_text(file_path, file_type, mime_type)
    
    async def _process_msg(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process MSG email files"""
        return await self._process_with_unstructured(file_path)
    
    # Log and data files
    async def _process_log(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process log files with pattern detection"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            lines = content.split('\n')
            
            # Analyze log patterns
            error_count = sum(1 for line in lines if 'ERROR' in line.upper())
            warning_count = sum(1 for line in lines if 'WARNING' in line.upper() or 'WARN' in line.upper())
            info_count = sum(1 for line in lines if 'INFO' in line.upper())
            
            # Create summary
            summary = f"Log file with {len(lines)} lines\n"
            summary += f"Errors: {error_count}\n"
            summary += f"Warnings: {warning_count}\n"
            summary += f"Info: {info_count}\n\n"
            
            # Add sample of errors and warnings
            errors = [line for line in lines if 'ERROR' in line.upper()][:5]
            if errors:
                summary += "Sample errors:\n" + "\n".join(errors) + "\n\n"
            
            warnings = [line for line in lines if 'WARNING' in line.upper() or 'WARN' in line.upper()][:5]
            if warnings:
                summary += "Sample warnings:\n" + "\n".join(warnings)
            
            documents = [
                Document(
                    text=content,
                    metadata={
                        'extraction_method': 'log_full',
                        'line_count': len(lines),
                        'error_count': error_count,
                        'warning_count': warning_count,
                        'info_count': info_count
                    }
                ),
                Document(
                    text=summary,
                    metadata={
                        'extraction_method': 'log_summary',
                        'analysis_type': 'error_analysis'
                    }
                )
            ]
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing log {file_path}: {e}")
            return await self._process_text(file_path, file_type, mime_type)
    
    async def _process_pcap(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process network capture files"""
        try:
            # This would require pyshark or scapy
            # For now, provide basic file info
            file_size = Path(file_path).stat().st_size
            
            return [Document(
                text=f"Network capture file: {file_path}\nSize: {file_size} bytes\nProcessing requires specialized tools.",
                metadata={
                    'extraction_method': 'pcap_basic',
                    'file_size': file_size
                }
            )]
        except Exception as e:
            logger.error(f"Error processing PCAP {file_path}: {e}")
            return []
    
    async def _process_vcf(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process vCard files"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Basic vCard parsing
            contacts = content.split('END:VCARD')
            contact_count = len([c for c in contacts if 'BEGIN:VCARD' in c])
            
            return [Document(
                text=content,
                metadata={
                    'extraction_method': 'vcf_raw',
                    'contact_count': contact_count
                }
            )]
            
        except Exception as e:
            logger.error(f"Error processing VCF {file_path}: {e}")
            return await self._process_text(file_path, file_type, mime_type)
    
    async def _process_ics(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process iCalendar files"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Basic iCal parsing
            events = content.split('END:VEVENT')
            event_count = len([e for e in events if 'BEGIN:VEVENT' in e])
            
            return [Document(
                text=content,
                metadata={
                    'extraction_method': 'ics_raw',
                    'event_count': event_count
                }
            )]
            
        except Exception as e:
            logger.error(f"Error processing ICS {file_path}: {e}")
            return await self._process_text(file_path, file_type, mime_type)
    
    # Archive processors
    async def _process_archive(self, file_path: str, file_type: str, mime_type: str) -> List[Document]:
        """Process archive files with support for more formats"""
        try:
            import zipfile
            import tarfile
            
            documents = []
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract based on type
                if file_type == 'zip':
                    with zipfile.ZipFile(file_path, 'r') as archive:
                        archive.extractall(temp_dir)
                elif file_type in ['tar', 'gz', 'bz2', 'xz']:
                    mode = 'r:*'  # Auto-detect compression
                    with tarfile.open(file_path, mode) as archive:
                        archive.extractall(temp_dir)
                elif file_type == 'rar':
                    try:
                        import rarfile
                        with rarfile.RarFile(file_path, 'r') as archive:
                            archive.extractall(temp_dir)
                    except ImportError:
                        logger.warning("RAR support not available")
                        return []
                elif file_type == '7z':
                    try:
                        import py7zr
                        with py7zr.SevenZipFile(file_path, 'r') as archive:
                            archive.extractall(temp_dir)
                    except ImportError:
                        logger.warning("7z support not available")
                        return []
                
                # Process extracted files
                temp_path = Path(temp_dir)
                for extracted_file in temp_path.rglob('*'):
                    if extracted_file.is_file() and extracted_file.stat().st_size < self.config.max_file_size_mb * 1024 * 1024:
                        try:
                            file_docs = await self.process_file(str(extracted_file))
                            for doc in file_docs:
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
        
        # Try flat reader first
        try:
            documents = await asyncio.to_thread(self.flat_reader.load_data, file_path)
            if documents:
                return documents
        except:
            pass
        
        # Try to process as text
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
            'processors': list(self.processors.keys()),
            'readers': {
                'document_readers': ['PDFReader', 'PyMuPDFReader', 'DocxReader', 'HWPReader', 'PptxReader', 'RTFReader'],
                'spreadsheet_readers': ['CSVReader', 'PandasCSVReader', 'PagedCSVReader'],
                'image_readers': ['ImageReader', 'ImageCaptionReader', 'ImageVisionLLMReader', 'ImageTabularChartReader'],
                'ebook_readers': ['EpubReader'],
                'email_readers': ['MboxReader'],
                'other_readers': ['IPYNBReader', 'XMLReader', 'VideoAudioReader', 'FlatReader', 'UnstructuredReader']
            }
        }