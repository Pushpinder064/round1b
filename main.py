#!/usr/bin/env python3
"""
Adobe India Hackathon 2025 - Round 1B
Persona-Driven Document Intelligence System
Complete implementation in a single file
"""
import nltk
nltk.download('punkt_tab')

import json
import re
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK data if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PersonaDocumentAnalyzer:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

        self.domain_keywords = {
            'academic_research': [
                'research', 'study', 'analysis', 'methodology', 'literature', 'hypothesis',
                'experiment', 'dataset', 'benchmark', 'algorithm', 'model', 'evaluation',
                'performance', 'results', 'conclusion', 'findings', 'investigation'
            ],
            'business_finance': [
                'revenue', 'profit', 'investment', 'market', 'strategy', 'growth',
                'financial', 'economic', 'business', 'company', 'corporate', 'industry',
                'competitive', 'analysis', 'trends', 'positioning', 'management'
            ],
            'education_learning': [
                'learning', 'education', 'student', 'study', 'knowledge', 'concept',
                'understanding', 'skills', 'preparation', 'exam', 'chapter', 'theory',
                'practice', 'exercise', 'explanation', 'mechanism', 'process'
            ],
            'technical_engineering': [
                'technical', 'engineering', 'system', 'design', 'implementation',
                'development', 'architecture', 'optimization', 'performance', 'efficiency',
                'solution', 'innovation', 'technology', 'application', 'framework'
            ],
            'healthcare_medical': [
                'medical', 'health', 'patient', 'treatment', 'diagnosis', 'clinical',
                'therapy', 'healthcare', 'medicine', 'disease', 'symptoms', 'care'
            ]
        }

    def extract_document_sections(self, pdf_path: str) -> List[Dict]:
        try:
            doc = fitz.open(pdf_path)
            sections = []
            current_section = None
            section_content = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")

                for block in blocks['blocks']:
                    if 'lines' not in block:
                        continue

                    for line in block['lines']:
                        line_text = ""
                        font_size = 0
                        is_bold = False

                        for span in line['spans']:
                            line_text += span['text']
                            font_size = max(font_size, span.get('size', 0))
                            is_bold = is_bold or (span.get('flags', 0) & 2**4)  # Bold flag

                        line_text = line_text.strip()
                        if not line_text:
                            continue

                        if self._is_section_heading(line_text, font_size, is_bold):
                            if current_section and section_content:
                                sections.append({
                                    'document': Path(pdf_path).name,
                                    'section_title': current_section,
                                    'content': ' '.join(section_content),
                                    'page_number': page_num + 1
                                })

                            current_section = line_text[:200]
                            section_content = []
                        else:
                            if current_section:
                                section_content.append(line_text)

                # Handle last section on page
                if current_section and section_content:
                    sections.append({
                        'document': Path(pdf_path).name,
                        'section_title': current_section,
                        'content': ' '.join(section_content),
                        'page_number': page_num + 1
                    })
                    current_section = None
                    section_content = []

            doc.close()
            logger.debug(f"Extracted {len(sections)} sections from {pdf_path}")
            return sections
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return []

    def _is_section_heading(self, text: str, font_size: float, is_bold: bool) -> bool:
        if len(text) < 3 or len(text) > 150:
            return False
        heading_patterns = [
            r'^\d+\.?\s+',
            r'^\d+\.\d+\.?\s+',
            r'^[A-Z][a-z]+\s+\d+',
            r'^[A-Z\s]{3,}$',
            r'^[A-Z][^.!?]*$',
        ]
        for pattern in heading_patterns:
            if re.match(pattern, text):
                return True
        if font_size > 12 and is_bold and len(text.split()) <= 8:
            return True
        words = text.split()
        if len(words) <= 6:
            capitalized_ratio = sum(1 for word in words if word[0].isupper()) / len(words)
            if capitalized_ratio >= 0.5:
                return True
        return False

    def analyze_persona_domains(self, persona: str, job: str) -> List[str]:
        combined_text = f"{persona} {job}".lower()
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                domain_scores[domain] = score
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        return [domain for domain, _ in sorted_domains[:3]]

    def score_section_relevance(self, section: Dict, persona: str, job: str,
                                relevant_domains: List[str]) -> float:
        content = f"{section['section_title']} {section['content']}"
        content_lower = content.lower()

        job_keywords = self._extract_keywords(job)
        job_matches = sum(1 for keyword in job_keywords if keyword in content_lower)
        job_score = min(job_matches / max(len(job_keywords), 1), 1.0)

        domain_score = 0
        for domain in relevant_domains:
            if domain in self.domain_keywords:
                domain_keywords = self.domain_keywords[domain]
                domain_matches = sum(1 for keyword in domain_keywords if keyword in content_lower)
                domain_score += domain_matches / len(domain_keywords)
        domain_score = min(domain_score, 1.0)

        quality_indicators = [
            'method', 'approach', 'result', 'analysis', 'conclusion', 'summary',
            'example', 'data', 'evidence', 'finding', 'insight', 'trend'
        ]
        quality_matches = sum(1 for indicator in quality_indicators if indicator in content_lower)
        quality_score = min(quality_matches / len(quality_indicators), 1.0)

        important_titles = [
            'introduction', 'method', 'result', 'conclusion', 'summary', 'analysis',
            'overview', 'background', 'discussion', 'finding', 'trend', 'strategy'
        ]
        title_lower = section['section_title'].lower()
        title_matches = sum(1 for title in important_titles if title in title_lower)
        title_score = min(title_matches / len(important_titles), 1.0)

        final_score = (
            job_score * 0.4 +
            domain_score * 0.3 +
            quality_score * 0.2 +
            title_score * 0.1
        )

        return final_score

    def _extract_keywords(self, text: str, max_keywords: int = 15) -> List[str]:
        words = word_tokenize(text.lower())
        keywords = [
            word for word in words
            if word not in self.stop_words and len(word) > 2 and word.isalpha()
        ]
        return list(set(keywords))[:max_keywords]

    def refine_subsection_content(self, sections: List[Dict], job: str,
                                 max_length: int = 400) -> List[Dict]:
        refined_subsections = []
        job_keywords = set(self._extract_keywords(job))

        for section in sections[:10]:
            content = section['content']
            sentences = sent_tokenize(content)

            if not sentences:
                continue

            sentence_scores = []
            for i, sentence in enumerate(sentences):
                if len(sentence.split()) < 5:
                    continue
                sentence_keywords = set(self._extract_keywords(sentence))
                overlap = len(sentence_keywords.intersection(job_keywords))
                overlap_score = overlap / max(len(job_keywords), 1) if job_keywords else 0
                position_score = 1.0 - (i / len(sentences)) * 0.4
                length_score = min(len(sentence.split()) / 25, 1.0)
                total_score = overlap_score * 0.5 + position_score * 0.3 + length_score * 0.2
                sentence_scores.append((sentence, total_score))

            sentence_scores.sort(key=lambda x: x[1], reverse=True)

            refined_content = ""
            for sentence, score in sentence_scores:
                if len(refined_content) + len(sentence) + 1 <= max_length:
                    refined_content += sentence + " "
                    if len(sentence_scores) <= 3:
                        continue
                else:
                    break

            if refined_content.strip():
                refined_subsections.append({
                    'document': section['document'],
                    'refined_text': refined_content.strip(),
                    'page_number': section['page_number']
                })

        return refined_subsections

    def process_collection(self, input_config_path: str) -> Dict[str, Any]:
        try:
            with open(input_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            challenge_info = config.get('challenge_info', {})
            documents = config.get('documents', [])
            persona = config.get('persona', {}).get('role', '')
            job_to_be_done = config.get('job_to_be_done', {}).get('task', '')

            logger.info(f"Processing challenge: {challenge_info.get('challenge_id', 'unknown')}")
            logger.info(f"Number of documents: {len(documents)}")
            logger.info(f"Persona: {persona}")
            logger.info(f"Job to be done: {job_to_be_done}")

            relevant_domains = self.analyze_persona_domains(persona, job_to_be_done)
            logger.debug(f"Relevant domains: {relevant_domains}")

            all_sections = []
            input_dir = Path(input_config_path).parent / "PDFs"
            logger.debug(f"Looking for PDFs in directory: {input_dir}")

            for doc_info in documents:
                pdf_path = input_dir / doc_info['filename']
                logger.debug(f"Checking PDF file: {pdf_path}")
                if not pdf_path.exists():
                    logger.warning(f"PDF not found: {pdf_path}")
                    continue
                sections = self.extract_document_sections(str(pdf_path))
                all_sections.extend(sections)

            if not all_sections:
                logger.warning("No sections extracted from PDFs.")
                return self._create_empty_output(config)

            scored_sections = []
            for section in all_sections:
                relevance_score = self.score_section_relevance(section, persona, job_to_be_done, relevant_domains)
                if relevance_score > 0.05:
                    scored_sections.append({**section, 'relevance_score': relevance_score})

            scored_sections.sort(key=lambda x: x['relevance_score'], reverse=True)

            extracted_sections = []
            for i, section in enumerate(scored_sections[:20]):
                extracted_sections.append({
                    'document': section['document'],
                    'section_title': section['section_title'],
                    'importance_rank': i + 1,
                    'page_number': section['page_number']
                })

            subsection_analysis = self.refine_subsection_content(scored_sections, job_to_be_done)

            output = {
                'metadata': {
                    'input_documents': [doc['filename'] for doc in documents],
                    'persona': persona,
                    'job_to_be_done': job_to_be_done,
                    'processing_timestamp': datetime.now().isoformat() + 'Z'
                },
                'extracted_sections': extracted_sections,
                'subsection_analysis': subsection_analysis
            }

            logger.info("Processing complete.")
            return output

        except Exception as e:
            logger.error(f"Error processing collection: {str(e)}")
            return self._create_empty_output(config if 'config' in locals() else {})

    def _create_empty_output(self, config: Dict) -> Dict[str, Any]:
        return {
            'metadata': {
                'input_documents': [],
                'persona': config.get('persona', {}).get('role', ''),
                'job_to_be_done': config.get('job_to_be_done', {}).get('task', ''),
                'processing_timestamp': datetime.now().isoformat() + 'Z'
            },
            'extracted_sections': [],
            'subsection_analysis': []
        }


def main():
    input_dir = Path("input")
    output_dir = Path("output")

    logger.info(f"Input directory exists: {input_dir.exists()}")
    if input_dir.exists():
        logger.info(f"Input directory contents:")
        for item in input_dir.iterdir():
            logger.info(f" - {item.name} (dir: {item.is_dir()})")
    else:
        logger.error("Input directory does not exist!")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = PersonaDocumentAnalyzer()

    collection_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    if not collection_dirs:
        logger.warning("No collection directories found in input folder")
        return

    logger.info(f"Found {len(collection_dirs)} collection(s) to process.")

    for collection_dir in collection_dirs:
        input_file = collection_dir / "challenge1b_input.json"
        logger.info(f"Looking for input JSON file: {input_file}, exists: {input_file.exists()}")
        if not input_file.exists():
            logger.warning(f"No input file found in {collection_dir.name}, skipping.")
            continue

        start_time = time.time()
        logger.info(f"Processing collection: {collection_dir.name}")

        result = analyzer.process_collection(str(input_file))

        output_file = output_dir / f"{collection_dir.name}_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Output saved to {output_file}")

        logger.info(f"Completed {collection_dir.name} in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
