# app.py

import os
from collections import OrderedDict
import json # if not already explicitly imported for loads/dumps
import time
import ntpath
import traceback
import base64
import io # For in-memory WAV saving
import sys # For exiting on critical errors

from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error, IntegrityError # Import IntegrityError
import numpy as np # Required by TTS for audio data
from scipy.io.wavfile import write as write_wav # To create WAV bytes

# --- Security ---
from werkzeug.security import generate_password_hash, check_password_hash

# --- ChromaDB & Text Splitting ---
import chromadb
from chromadb.config import Settings # Optional for config
from langchain.text_splitter import RecursiveCharacterTextSplitter


# --- Import project modules ---
# Ensure config.py exists and defines the necessary variables
try:
    from config import (
        OUTPUT_DIR, DB_CONFIG, GROBID_API, PDFFIGURES_JAR_PATH,
        CHROMA_DB_PATH, CHROMA_COLLECTION_NAME,
        GEMINI_API_KEY, GEMINI_MODEL_NAME # <-- IMPORT GEMINI CONFIG
    )
    print("Successfully imported configuration variables from config.py")
except ImportError as e:
    # Check specifically for Gemini keys if other imports worked
    gemini_keys_imported = False
    try:
        from config import GEMINI_API_KEY, GEMINI_MODEL_NAME
        gemini_keys_imported = True
    except ImportError:
        print("WARNING: GEMINI_API_KEY or GEMINI_MODEL_NAME not found in config.py. Course/Quiz generation will be unavailable.")
        GEMINI_API_KEY = None # Ensure they are None if import fails
        GEMINI_MODEL_NAME = None
    
    # Report the original error, but mention Gemini status
    print(f"WARNING: Could not import all expected variables from config.py. Error: {e}")
    if not gemini_keys_imported:
        print("       (Gemini API Key/Model Name also missing or failed to import)")
    # Decide if you want to exit or just warn
    # sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during config import: {e}")
    traceback.print_exc()
    sys.exit(1)

from utils import get_db_connection
# Import model loading functions
from model_loader import (
    load_qa_model, load_summarization_model, load_embedding_model, load_cross_encoder_model
)
# Import Agent classes
from extraction_agent import ExtractionAgentLangGraph
from summarization_agent import SummarizationAgent
from qa_agent import QuestionAnsweringAgent
from translation_agent import TranslationAgent
from course_quiz_agent import CourseQuizGenerationAgent # <-- IMPORT NEW AGENT
from multiplecourse_agent import AdvancedCourseGeneratorAgent # <-- IMPORT NEW AGENT

# --- NEW: Import TTS ---
try:
    import torch
    from TTS.api import TTS
    TTS_AVAILABLE = True
    print("Coqui TTS library found.")
except ImportError:
    TTS_AVAILABLE = False
    TTS = None
    torch = None
    print("WARNING: Coqui TTS library ('TTS') not found or import failed. TTS feature will be unavailable.")
    print("Install with: pip install TTS")

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)
app.config['JSON_SORT_KEYS'] = False  # <<< MAKE SURE THIS IS SET AND EFFECTIVE

# --- Configure App Settings from Imported Variables ---
try:
    if not OUTPUT_DIR or not isinstance(OUTPUT_DIR, str):
        raise ValueError("OUTPUT_DIR from config.py is invalid or not defined.")
    app.config['OUTPUT_DIR'] = os.path.abspath(OUTPUT_DIR)
    os.makedirs(app.config['OUTPUT_DIR'], exist_ok=True)
    print(f"Output directory configured: {app.config['OUTPUT_DIR']}")
except Exception as e:
    print(f"FATAL ERROR configuring OUTPUT_DIR: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    if not CHROMA_DB_PATH or not isinstance(CHROMA_DB_PATH, str):
        raise ValueError("CHROMA_DB_PATH from config.py is invalid or not defined.")
    app.config['CHROMA_DB_PATH'] = os.path.abspath(CHROMA_DB_PATH)
    os.makedirs(app.config['CHROMA_DB_PATH'], exist_ok=True)
    print(f"ChromaDB persistent path configured: {app.config['CHROMA_DB_PATH']}")
except Exception as e:
    print(f"FATAL ERROR configuring CHROMA_DB_PATH: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Global Variables for Models & Pipelines ---
qa_pipeline_instance = None
qa_tokenizer = None
qa_pipeline_device_name = "Not Initialized"
summarizer_pipeline_instance = None
summarizer_device_name = "Not Initialized"
embedding_model_instance = None
embedding_tokenizer = None
embedding_device = "cpu"
embedding_model_device_name = "Not Initialized"
cross_encoder_model_instance = None
cross_encoder_device = "cpu"
cross_encoder_device_name = "Not Initialized"
translation_agent = None
course_quiz_agent = None # <-- ADD GLOBAL FOR NEW AGENT
advanced_course_agent = None # <-- NEW AGENT GLOBAL (for both single and multi-article)


# --- TTS Global Variables ---
tts_instance = None
tts_device = "cpu"
tts_model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
tts_speakers = []
tts_languages = []
tts_status = "Not Initialized"

# --- ChromaDB Client ---
chroma_client = None
chroma_collection = None

# --- Database Setup ---
def create_tables_if_not_exist():
    """Creates/Alters tables: adds generated_course & generated_quiz to articles."""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if not conn:
            print("ERROR: Could not connect to database to verify tables.")
            return False
        cursor = conn.cursor()

        # --- Create articles table statement (ensure it matches your structure) ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INT AUTO_INCREMENT PRIMARY KEY,
                title TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
                sections LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL DEFAULT NULL,
                Insights TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
                figures LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL DEFAULT NULL,
                tables LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                Pages INT NULL DEFAULT 0
                -- New columns will be added below if they don't exist
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)
        print("Checked/Created 'articles' table base structure.")

        # --- Add generated_course column if it doesn't exist ---
        try:
            # Use LONGTEXT for potentially large generated content
            # Using utf8mb4_general_ci for general text comparison/sorting if needed
            cursor.execute("""
                ALTER TABLE articles
                ADD COLUMN IF NOT EXISTS generated_course LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT 'Course outline generated by AI'
                AFTER Pages;
            """)
            print("Checked/Added 'generated_course' column to 'articles' table.")
        except Error as alter_err:
            # Ignore "Duplicate column name" error (1060), raise others
            if alter_err.errno == 1060:
                 print("Column 'generated_course' already exists.")
            else:
                 print(f"Error altering table for generated_course: {alter_err}")
                 raise alter_err # Re-raise other alter errors

        # --- Add generated_quiz column if it doesn't exist ---
        try:
            cursor.execute("""
                ALTER TABLE articles
                ADD COLUMN IF NOT EXISTS generated_quiz LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT 'Quiz generated by AI based on the course'
                AFTER generated_course;
            """)
            print("Checked/Added 'generated_quiz' column to 'articles' table.")
        except Error as alter_err:
            if alter_err.errno == 1060:
                print("Column 'generated_quiz' already exists.")
            else:
                print(f"Error altering table for generated_quiz: {alter_err}")
                raise alter_err

        # --- Create users table statement ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(120) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                recently_analyzed_ids TEXT NULL COMMENT 'JSON array of article IDs',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)
        print("Checked/Created 'users' table.")

        conn.commit()
        return True
    except Error as e:
        print(f"Database error during table creation/alteration: {e}")
        if conn: conn.rollback()
        return False
    except Exception as e:
        print(f"Unexpected error during table creation/alteration: {e}")
        traceback.print_exc()
        if conn: conn.rollback()
        return False
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()

# --- Load Models, Initialize ChromaDB, and Initialize Agents ---
print("\n" + "="*30 + " Initializing Components " + "="*30)
print("\n--- Verifying Database Tables (Including Course/Quiz Columns) ---")
if not create_tables_if_not_exist():
    print("WARNING: Failed to verify/create/alter database tables. Database operations might fail.")
else:
    print("Database tables verified/updated successfully.")

print("\n" + "="*30 + " Initializing Models, Vector DB & Agents " + "="*30)
try:
    # --- Load ML Models ---
    print("\n--- Loading Machine Learning Models ---")
    qa_pipeline_instance, qa_tokenizer, qa_pipeline_device_name = load_qa_model()
    summarizer_pipeline_instance, summarizer_device_name = load_summarization_model()
    embedding_model_instance, embedding_tokenizer, embedding_device, embedding_model_device_name = load_embedding_model()
    cross_encoder_model_instance, cross_encoder_device, cross_encoder_device_name = load_cross_encoder_model()

    # --- Load TTS Model ---
    print("\n--- Loading Text-to-Speech Model ---")
    if TTS_AVAILABLE and TTS and torch:
        try:
            tts_device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Attempting to load TTS model '{tts_model_name}' onto device: {tts_device}")
            tts_instance = TTS(model_name=tts_model_name).to(tts_device)
            if hasattr(tts_instance, 'speakers') and tts_instance.speakers:
                 tts_speakers = tts_instance.speakers
            if hasattr(tts_instance, 'languages') and tts_instance.languages:
                 tts_languages = tts_instance.languages
            tts_status = f"Available (on {tts_device})"
            print(f"TTS Model loaded successfully. Device: {tts_device}")
        except Exception as e_tts:
            print(f"ERROR loading TTS model: {e_tts}")
            traceback.print_exc()
            tts_instance = None
            tts_status = f"Failed to load: {type(e_tts).__name__}"
    else:
        print("Skipping TTS model loading (library not available).")
        tts_status = "Unavailable (Library Missing)"

    # --- Initialize ChromaDB ---
    print("\n--- Initializing ChromaDB Client ---")
    try:
        chroma_db_path_to_use = app.config.get('CHROMA_DB_PATH')
        if not chroma_db_path_to_use:
             raise ValueError("CHROMA_DB_PATH not found in Flask app config.")
        chroma_client = chromadb.PersistentClient(
            path=chroma_db_path_to_use,
            settings=Settings(anonymized_telemetry=False) # Disable telemetry
        )
        print(f"ChromaDB client initialized. Persistent path: {chroma_db_path_to_use}")
        # Get or create the collection
        chroma_collection = chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME
        )
        print(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' accessed/created.")
        print(f"Initial document count in collection: {chroma_collection.count()}")
    except Exception as e_chroma_init:
        print(f"ERROR initializing ChromaDB client or collection: {e_chroma_init}")
        traceback.print_exc()
        chroma_client = None
        chroma_collection = None

    # --- Instantiate Agents ---
    print("\n--- Instantiating Agents ---")
    if not GROBID_API or not PDFFIGURES_JAR_PATH:
         raise ValueError("GROBID_API or PDFFIGURES_JAR_PATH not correctly defined in config.")
    extraction_agent = ExtractionAgentLangGraph(GROBID_API, PDFFIGURES_JAR_PATH, OUTPUT_DIR) # Pass output dir if needed
    summarization_agent = SummarizationAgent(
        # pipeline_instance=summarizer_pipeline_instance,
        # device_name=summarizer_device_name
    )
    qa_agent = QuestionAnsweringAgent(
        embedding_model=embedding_model_instance,
        cross_encoder_model=cross_encoder_model_instance,
        qa_pipeline=qa_pipeline_instance, # Keep even if not primary path
        chroma_collection=chroma_collection,
        embedding_device=embedding_device,
        cross_encoder_device=cross_encoder_device
    )
    print(f"QA Agent Initialized. Chroma Status: {'Available' if qa_agent.is_chroma_available else 'Unavailable'}")

    print("\n--- Initializing Translation Agent (googletrans) ---")
    translation_agent = TranslationAgent()

    # --- MODIFIED: Initialize AdvancedCourseGeneratorAgent ---
    print("\n--- Initializing Advanced Course Generator Agent (Gemini) ---")
    if GEMINI_API_KEY and GEMINI_MODEL_NAME:
        advanced_course_agent = AdvancedCourseGeneratorAgent( # Use the new agent
            api_key=GEMINI_API_KEY,
            model_name=GEMINI_MODEL_NAME
        )
        print(f"Advanced Course Agent Initialized Status: {'Available' if advanced_course_agent.is_available else 'Failed (Check API Key/Model/Logs)'}")
    else:
        print("Advanced Course Agent: Skipped initialization (GEMINI_API_KEY or GEMINI_MODEL_NAME not found/invalid in config/env).")
        advanced_course_agent = None # Ensure it's None
    # --- Initialize Course/Quiz Agent --- <--- ADDED THIS BLOCK ---
    print("\n--- Initializing Course & Quiz Generation Agent (Gemini) ---")
    if GEMINI_API_KEY: # Only initialize if the key was loaded/imported
        course_quiz_agent = CourseQuizGenerationAgent(
            api_key=GEMINI_API_KEY,
            model_name=GEMINI_MODEL_NAME
        )
        print(f"Course/Quiz Agent Initialized Status: {'Available' if course_quiz_agent.is_available else 'Failed (Check API Key/Logs)'}")
    else:
        print("Course/Quiz Agent: Skipped initialization (GEMINI_API_KEY not found or invalid in config/env).")
        course_quiz_agent = None # Ensure it's None if key is missing
    # --- END OF NEW BLOCK ---

    print("="*90)
    print("--- Initialization Complete ---")

except Exception as E:
    print(f"FATAL ERROR DURING INITIALIZATION BLOCK: {E}")
    traceback.print_exc()
    print("Application might be in an unstable state.")
    # sys.exit(1) # Optional: Exit if initialization fails critically


# --- Helper Function for Token Length ---
def count_tokens(text: str) -> int:
    """Counts tokens using the globally loaded embedding tokenizer."""
    if embedding_tokenizer:
        try:
            # Use encode which typically returns a list of token IDs
            return len(embedding_tokenizer.encode(text))
        except Exception as e:
             # Fallback to character count if tokenizer fails
             print(f"Warning: Tokenizer encode failed ({e}), falling back to character count for length.")
             return len(text)
    else:
        # Fallback if no tokenizer is loaded
        return len(text)


# --- PDF Processing Route ---
# --- PDF Processing Route ---
@app.route('/api/process-pdf', methods=['POST'])
def process_pdf():
    start_time_route = time.time()
    print("\n--- Received request for /api/process-pdf ---")
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file part in the request"}), 400
    pdf_file = request.files['pdf']
    if not pdf_file or not pdf_file.filename:
        return jsonify({"error": "No selected PDF file"}), 400
    if not pdf_file.filename.lower().endswith('.pdf'):
         return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

    try:
        safe_basename = "".join(c if c.isalnum() or c in ['_', '.', '-'] else '_' for c in ntpath.basename(pdf_file.filename))
        output_dir_config = app.config.get('OUTPUT_DIR') # Use app.config
        if not output_dir_config:
             print("CRITICAL ERROR: OUTPUT_DIR not found in Flask config.")
             return jsonify({"error": "Server configuration error (output path)."}), 500
        temp_pdf_path = os.path.join(output_dir_config, f"upload_{int(time.time())}_{os.getpid()}_{safe_basename}")
    except Exception as path_err:
        print(f"Error creating temporary file path: {path_err}")
        return jsonify({"error": "Server error preparing file storage."}), 500

    conn = None
    cursor = None
    article_title = "Unknown Title"
    extracted_data = None
    inserted_id = None
    page_count = 0
    insights_summary = "[Insights summary not generated]"

    # --- Helper to convert sections OrderedDict/dict to list of dicts for frontend ---
    def format_sections_for_frontend(sections_data_internal):
        if not sections_data_internal: # Handle None or empty
            return []
        
        formatted_sections = []
        if isinstance(sections_data_internal, OrderedDict):
            for h, c_val in sections_data_internal.items():
                if isinstance(c_val, dict):
                    # Ensure all expected keys exist, default to null or empty string
                    section_item = {
                        "heading": h,
                        "summary": c_val.get("summary"), # Default to None if missing
                        "full_text": c_val.get("full_text", "") # Default to empty string if missing
                    }
                    formatted_sections.append(section_item)
                else: # If content is not a dict, try to adapt
                    print(f"Warning: Section content for '{h}' is not a dict. Adapting. Value: {str(c_val)[:50]}")
                    formatted_sections.append({"heading": h, "summary": None, "full_text": str(c_val)})
            return formatted_sections
        elif isinstance(sections_data_internal, dict): # Regular dict
            print("Warning: sections_data_internal was a dict, not OrderedDict. Frontend order might be alphabetical.")
            # Sort by key for some consistency for regular dicts.
            # Ideally, the source (ExtractionAgent) should provide an OrderedDict.
            for h, c_val in sorted(sections_data_internal.items()):
                if isinstance(c_val, dict):
                    section_item = {
                        "heading": h,
                        "summary": c_val.get("summary"),
                        "full_text": c_val.get("full_text", "")
                    }
                    formatted_sections.append(section_item)
                else:
                    print(f"Warning: Section content for '{h}' is not a dict (in regular dict). Adapting. Value: {str(c_val)[:50]}")
                    formatted_sections.append({"heading": h, "summary": None, "full_text": str(c_val)})
            return formatted_sections
        elif isinstance(sections_data_internal, list): # Already in desired list format
             # Validate structure of list items
            for item_idx, item in enumerate(sections_data_internal):
                if isinstance(item, dict) and "heading" in item:
                    # Ensure summary and full_text exist
                    item["summary"] = item.get("summary") 
                    item["full_text"] = item.get("full_text", "")
                    formatted_sections.append(item)
                else:
                    print(f"Warning: Item at index {item_idx} in sections list is not a valid section object. Skipping.")
            return formatted_sections
        print(f"Warning: Unhandled sections data type: {type(sections_data_internal)}. Returning empty list.")
        return []

    try:
        pdf_file.save(temp_pdf_path)
        print(f"PDF saved temporarily to: {temp_pdf_path}")

        global extraction_agent # Assuming extraction_agent is global
        if not extraction_agent:
             print("Error: Extraction agent not initialized globally.")
             return jsonify({"error": "Extraction service unavailable."}), 503
        
        extracted_data = extraction_agent.extract(temp_pdf_path)
        if not extracted_data:
            print("Error: Extraction Agent returned None.")
            return jsonify({"error": "Failed to extract any data from the PDF."}), 500
        
        page_count = extracted_data.get('page_count', 0)
        article_title = extracted_data.get("title", "Unknown Title")
        print(f"Extracted Title: '{article_title[:100]}...'")

        if not article_title or article_title in ["No title found", "Extraction Error Title", "Parsing Error Title", "GROBID Failed", "GROBID Parsing Failed"]:
             return jsonify({"error": "Failed to extract required information (like title) from the PDF structure."}), 422
        
        # Ensure sections is at least an empty OrderedDict if missing from extraction
        # CRITICAL: extraction_agent.extract SHOULD return OrderedDict for sections if order is from document
        if "sections" not in extracted_data or not isinstance(extracted_data["sections"], (dict, OrderedDict)):
            print("Warning: extracted_data['sections'] from extraction agent is missing or not a dict/OrderedDict. Initializing empty OrderedDict.")
            extracted_data["sections"] = OrderedDict()
        elif not isinstance(extracted_data["sections"], OrderedDict):
            # If it's a regular dict, we make it OrderedDict. Order depends on Python version & how dict was built.
            print("Warning: extracted_data['sections'] from extraction agent was a dict, not OrderedDict. Converting. Document order relies on agent.")
            extracted_data["sections"] = OrderedDict(extracted_data["sections"])

        if "figures" not in extracted_data: extracted_data["figures"] = []
        if "tables" not in extracted_data: extracted_data["tables"] = []

        # Check Database for Existing Article
        conn = get_db_connection()
        existing_article_data = None
        if conn:
            cursor = conn.cursor(dictionary=True)
            query = "SELECT id, title, sections, figures, tables, Insights, Pages, generated_course, generated_quiz FROM articles WHERE title = %s"
            cursor.execute(query, (article_title,))
            existing_article_data = cursor.fetchone()
            if cursor: cursor.close()
            if conn.is_connected(): conn.close()
            conn = None; cursor = None
        else:
            print("Warning: DB connection failed for check.")

        if existing_article_data:
            print(f"Article '{article_title[:100]}...' found in DB. Returning existing data.")
            try:
                sections_str = existing_article_data.get('sections')
                sections_data_from_db = json.loads(sections_str, object_pairs_hook=OrderedDict) if sections_str else OrderedDict()
                
                figures_serializable = json.loads(existing_article_data.get('figures', '[]'))
                tables_serializable = json.loads(existing_article_data.get('tables', '[]'))
            except (json.JSONDecodeError, TypeError) as json_err:
                print(f"Warning: Error decoding JSON from DB for article {existing_article_data['id']}: {json_err}.")
                sections_data_from_db = OrderedDict()
                figures_serializable = []
                tables_serializable = []
            
            response_data = {
                "id": existing_article_data.get('id'),
                "title": existing_article_data.get('title'),
                "sections": format_sections_for_frontend(sections_data_from_db), # Format for frontend
                "figures": figures_serializable,
                "tables": tables_serializable,
                "insights": existing_article_data.get('Insights'),
                "pages": existing_article_data.get('Pages'),
                "generated_course": existing_article_data.get('generated_course'),
                "generated_quiz": existing_article_data.get('generated_quiz'),
                "message": "Retrieved from database",
                "status": "existing"
            }
            if os.path.exists(temp_pdf_path): os.remove(temp_pdf_path)
            return jsonify(response_data), 200

        # --- Process New PDF (Summarization) ---
        # extracted_data["sections"] should be an OrderedDict here
        global summarization_agent # Assuming global
        successful_section_summaries = []
        if summarization_agent:
            print("Starting section summarization...")
            sections_to_summarize = extracted_data.get('sections', OrderedDict()) # Default to empty OrderedDict
            if isinstance(sections_to_summarize, OrderedDict): # Ensure it's what we expect
                for section_heading, section_content_dict in list(sections_to_summarize.items()): # Iterate over OrderedDict
                    summary = "[Summarization Error]"
                    if isinstance(section_content_dict, dict) and section_content_dict.get("full_text", "").strip():
                        try:
                            summary = summarization_agent.summarize(section_content_dict["full_text"])
                            if not summary or 'failed' in summary.lower() or summary.strip().startswith("Error:"):
                                summary = "[Summarization Error]"
                            else:
                                successful_section_summaries.append(summary)
                        except Exception as sum_err:
                            print(f"Error summarizing section '{section_heading}': {sum_err}")
                            summary = "[Summarization Error]"
                    else: # Content not a dict or no full_text
                        summary = None # Or "[No text to summarize]"
                        print(f"Skipping summarization for section '{section_heading}' due to invalid content or missing text.")
                    
                    # Update the OrderedDict directly
                    if isinstance(sections_to_summarize.get(section_heading), dict):
                        sections_to_summarize[section_heading]['summary'] = summary
                    else: # If the section content wasn't a dict, create one
                        print(f"Warning: section content for '{section_heading}' was not a dict. Re-structuring for summary.")
                        sections_to_summarize[section_heading] = {"full_text": str(section_content_dict), "summary": summary}
            else:
                print(f"Warning: 'sections' data for summarization is not an OrderedDict. Type: {type(sections_to_summarize)}")
        else: # Summarization agent not available
            print("Warning: Summarization Agent unavailable. Skipping section summarization.")
            # Ensure 'summary' key exists in all section dicts in the OrderedDict
            sections_data = extracted_data.get('sections', OrderedDict())
            if isinstance(sections_data, OrderedDict):
                for heading, content in sections_data.items():
                    if isinstance(content, dict) and 'summary' not in content:
                        sections_data[heading]['summary'] = None # Or "[Summarization Skipped]"

        # Generate Final Insights Summary
        if summarization_agent and successful_section_summaries:
            combined_summary_text = "\n\n".join(successful_section_summaries)
            try:
                 insights_summary = summarization_agent.summarize(combined_summary_text)
                 if not insights_summary or 'failed' in insights_summary.lower() or 'error' in insights_summary.lower():
                      insights_summary = "[Insight generation failed]"
            except Exception: insights_summary = "[Insight generation error]"
        elif not successful_section_summaries: insights_summary = "[No sections summarized successfully]"
        else: insights_summary = "[Summarization service unavailable for insights]"
        print(f"Final Insights: '{insights_summary[:100]}...'")

        # Save to SQL Database
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            # Save extracted_data["sections"] (which is OrderedDict) as JSON string
            sections_json_db = json.dumps(extracted_data.get('sections', OrderedDict()))
            figures_json_db = json.dumps(extracted_data.get('figures', []))
            tables_json_db = json.dumps(extracted_data.get('tables', []))
            insert_query = """
                INSERT INTO articles (title, sections, Insights, figures, tables, Pages)
                VALUES (%s, %s, %s, %s, %s, %s)"""
            insert_data = (article_title, sections_json_db, insights_summary, figures_json_db, tables_json_db, page_count)
            cursor.execute(insert_query, insert_data)
            conn.commit()
            inserted_id = cursor.lastrowid
            print(f"Article saved to SQL DB with ID: {inserted_id}")
            if cursor: cursor.close()
            if conn.is_connected(): conn.close()
            conn = None; cursor = None
        else:
            print("Skipping SQL save (connection failed).")
            # If DB save fails, we might not want to proceed with Chroma if it depends on SQL ID
            if os.path.exists(temp_pdf_path): os.remove(temp_pdf_path)
            return jsonify({"error": "Failed to save article to database. Processing halted."}), 500

        # Chunk, Embed, and Save to ChromaDB (IF SQL Save Succeeded)
        # ... (Your existing ChromaDB logic, ensure it uses extracted_data['sections'] (OrderedDict) correctly)
        # For brevity, I'm omitting the full ChromaDB logic. It should be mostly fine if `extracted_data['sections']` is an OrderedDict.
        chroma_save_status = "Skipped (Logic Omitted for Brevity in this Snippet)"
        global chroma_collection, embedding_model_instance, embedding_tokenizer, embedding_device
        if chroma_collection and embedding_model_instance and embedding_tokenizer and inserted_id:
            # ... (Your ChromaDB processing logic here) ...
            # Ensure `section_texts_for_meta` is built from the OrderedDict `extracted_data['sections']`
            # to maintain order for metadata generation.
            pass # Placeholder for your ChromaDB logic
        else:
            print("Skipping ChromaDB processing due to missing components or no SQL ID.")


        final_response = {
            "id": inserted_id,
            "title": article_title,
            "sections": format_sections_for_frontend(extracted_data.get('sections', OrderedDict())), # Format for frontend
            "figures": extracted_data.get('figures', []),
            "tables": extracted_data.get('tables', []),
            "insights": insights_summary,
            "pages": page_count,
            "generated_course": None,
            "generated_quiz": None,
            "message": f"Processed successfully. SQL ID: {inserted_id}. Vector Store: {chroma_save_status}.",
            "chroma_status": chroma_save_status,
            "status": "created"
        }
        if os.path.exists(temp_pdf_path): os.remove(temp_pdf_path)
        return jsonify(final_response), 201

    except Exception as e:
        print(f"An unexpected error occurred in /api/process-pdf: {e}")
        traceback.print_exc()
        # ... (your existing generic error handling) ...
        if 'temp_pdf_path' in locals() and os.path.exists(temp_pdf_path):
             try: os.remove(temp_pdf_path)
             except OSError: pass
        return jsonify({"error": f"Internal server error processing PDF ({type(e).__name__})."}), 500
    finally:
        if 'temp_pdf_path' in locals() and os.path.exists(temp_pdf_path):
             try: os.remove(temp_pdf_path)
             except OSError as e_del: print(f"Error deleting temp PDF {temp_pdf_path}: {e_del}")
        if cursor:
            try: cursor.close()
            except Exception: pass
        if conn and conn.is_connected():
            try: conn.close()
            except Exception: pass
        print("-" * 70)

# --- Chat Route ---
@app.route('/api/chat', methods=['POST'])
def handle_chat():
    start_time_route = time.time()
    print("\n--- Received request for /api/chat ---")
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    user_question = data.get('message')
    context_id_str = data.get('context_id') # Expecting the SQL article ID

    # Validate input
    if not user_question or not isinstance(user_question, str) or not user_question.strip():
        return jsonify({"error": "Invalid or missing 'message'."}), 400
    if context_id_str is None:
         return jsonify({"error": "Missing 'context_id'."}), 400

    print(f"Received Context SQL ID: '{context_id_str}', Question: '{user_question[:100]}...'")

    # Validate context_id format
    try:
        context_db_id = int(context_id_str)
        if context_db_id < 0: raise ValueError("Context ID cannot be negative")
    except (ValueError, TypeError):
        return jsonify({"error": f"Invalid 'context_id': must be a non-negative integer."}), 400

    # --- Fetch Title from SQL (Optional, for context logging/display) ---
    article_title = "Unknown Title"
    conn = None; cursor = None
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            query = "SELECT title FROM articles WHERE id = %s"
            cursor.execute(query, (context_db_id,))
            article_data = cursor.fetchone()
            if article_data:
                article_title = article_data.get('title', 'Unknown Title')
            else:
                 print(f"Warning: Article ID {context_db_id} not found in DB for title lookup.")
                 return jsonify({"error": f"Document context with ID '{context_db_id}' not found."}), 404
        else:
            print("Warning: DB connection failed for title lookup.")
    except Error as e:
        print(f"Database error fetching title for article ID {context_db_id}: {e}")
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()
        conn = None; cursor = None

    print(f"Using Title: '{article_title[:100]}...' for context reference.")

    # --- Use QA Agent ---
    if not qa_agent or not qa_agent.is_ready:
        missing_details = "QA Agent not initialized"
        if qa_agent:
            missing_details = qa_agent.get_missing_components_message()
        print(f"Error: QA Agent not ready. {missing_details}")
        return jsonify({"error": f"The Question Answering service is currently unavailable. Details: {missing_details}"}), 503

    print(f"Invoking QA Agent for article ID {context_db_id}...")
    qa_start = time.time()
    try:
        # Pass the SQL database ID to the QA agent
        qa_response = qa_agent.answer(question=user_question, article_title=article_title, context_db_id=context_db_id)
        qa_end = time.time()
        print(f"QA Agent processing took {qa_end - qa_start:.2f} seconds.")

        if not isinstance(qa_response, dict):
            print(f"Error: QA Agent returned unexpected type: {type(qa_response)}")
            return jsonify({"error": "Invalid response structure from QA service."}), 500

    except Exception as agent_err:
         print(f"CRITICAL Error during qa_agent.answer(): {agent_err}")
         traceback.print_exc()
         err_detail = str(agent_err)
         if "chroma" in err_detail.lower():
             return jsonify({"error": "Error retrieving context from vector store."}), 500
         return jsonify({"error": f"An unexpected error occurred within the QA agent ({type(agent_err).__name__})."}), 500

    # --- Return Response ---
    status_code = 500 if "error" in qa_response else 200
    if status_code == 500:
        print(f"QA Agent returned error: {qa_response.get('error')}")
    else:
        print("QA Agent returned response successfully.")

    route_end_time = time.time()
    print(f"--- /api/chat request completed in {route_end_time - start_time_route:.2f} seconds ---")
    print("-" * 70)
    return jsonify(qa_response), status_code


# --- Translate Route ---
@app.route('/api/translate', methods=['POST'])
def handle_translate():
    start_time_route = time.time()
    print("\n--- Received request for /api/translate ---")
    if not translation_agent or not translation_agent.is_available:
        return jsonify({"error": "Translation service is currently unavailable."}), 503

    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    text_to_translate = data.get('text')
    target_language = data.get('target_language')
    source_language = data.get('source_language') # Optional

    if not text_to_translate or not isinstance(text_to_translate, str) or not text_to_translate.strip():
        return jsonify({"error": "Invalid or missing 'text' field."}), 400
    if not target_language or not isinstance(target_language, str) or not target_language.strip():
        return jsonify({"error": "Invalid or missing 'target_language' field."}), 400
    if source_language and not isinstance(source_language, str):
         return jsonify({"error": "Invalid 'source_language' field."}), 400

    print(f"Translate Request - Target: '{target_language}', Source: '{source_language or 'Auto'}', Text: '{text_to_translate[:100]}...'")

    agent_start_time = time.time()
    try:
        result = translation_agent.translate(
            text=text_to_translate,
            target_lang=target_language,
            source_lang=source_language
        )
        agent_end_time = time.time()
        print(f"Translation agent finished in {agent_end_time - agent_start_time:.2f} seconds.")

        if not isinstance(result, dict):
             print(f"Error: Translation agent returned unexpected type: {type(result)}")
             return jsonify({"error": "Received invalid response from translation service"}), 500

    except Exception as agent_err:
         print(f"CRITICAL Error during translation_agent.translate(): {agent_err}")
         traceback.print_exc()
         return jsonify({"error": "An unexpected error occurred within the Translation agent."}), 500

    status_code = 500 if "error" in result else 200
    if status_code == 500:
        print(f"Translation agent returned error: {result.get('error')} - {result.get('details')}")
    else:
        print("Translation successful.")

    route_end_time = time.time()
    print(f"--- /api/translate request completed in {route_end_time - start_time_route:.2f} seconds ---")
    print("-" * 70)
    return jsonify(result), status_code


# --- TTS Route ---
@app.route('/api/tts', methods=['POST'])
def handle_tts():
    start_time_route = time.time()
    print("\n--- Received request for /api/tts ---")

    if not tts_instance:
        error_detail = "TTS library missing" if not TTS_AVAILABLE else "Model failed to load"
        print(f"Error: TTS service unavailable ({error_detail}).")
        return jsonify({"error": "Text-to-Speech service unavailable.", "details": error_detail}), 503

    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    text_to_speak = data.get('text')
    language = data.get('language', 'en') # Default to English

    if not text_to_speak or not isinstance(text_to_speak, str) or not text_to_speak.strip():
        return jsonify({"error": "Invalid or missing 'text' field for TTS."}), 400

    selected_speaker = None
    if hasattr(tts_instance, 'speakers') and tts_instance.speakers:
        selected_speaker = tts_instance.speakers[0] # Use first available speaker

    print(f"TTS Request - Lang: '{language}', Speaker: '{selected_speaker or 'Default'}', Text: '{text_to_speak[:100]}...'")

    tts_start_time = time.time()
    try:
        print("Invoking TTS model...")
        wav_data = tts_instance.tts(text=text_to_speak, speaker=selected_speaker, language=language)

        if not isinstance(wav_data, (np.ndarray, list)):
             raise TypeError(f"TTS model returned invalid audio format: {type(wav_data)}")
        if isinstance(wav_data, list): wav_data = np.array(wav_data, dtype=np.float32)
        if wav_data.size == 0: raise ValueError("TTS returned empty audio data.")

        tts_end_time = time.time()
        print(f"TTS model generated audio data in {tts_end_time - tts_start_time:.2f} seconds.")

        conversion_start_time = time.time()
        sample_rate = 24000 # Default
        if hasattr(tts_instance, 'synthesizer') and hasattr(tts_instance.synthesizer, 'output_sample_rate'):
            sample_rate = tts_instance.synthesizer.output_sample_rate
        print(f"Using sample rate: {sample_rate}")

        if wav_data.dtype == np.float32 or wav_data.dtype == np.float64:
            max_val = np.max(np.abs(wav_data)); max_val = 1.0 if max_val == 0 else max_val
            wav_data_int16 = np.int16(wav_data / max_val * 32767)
        elif wav_data.dtype == np.int16:
            wav_data_int16 = wav_data
        else:
            print(f"Warning: Unexpected audio data type {wav_data.dtype}. Attempting conversion.")
            try:
                 float_data = wav_data.astype(np.float32); max_val = np.max(np.abs(float_data)); max_val = 1.0 if max_val == 0 else max_val
                 wav_data_int16 = np.int16(float_data / max_val * 32767)
            except Exception as conv_err:
                 raise ValueError(f"Could not convert audio data type {wav_data.dtype} to int16") from conv_err

        wav_buffer = io.BytesIO()
        write_wav(wav_buffer, sample_rate, wav_data_int16)
        wav_buffer.seek(0); wav_bytes = wav_buffer.read(); wav_buffer.close()
        conversion_end_time = time.time()
        print(f"Converted audio to WAV in {conversion_end_time - conversion_start_time:.2f} seconds.")

        audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')

        route_end_time = time.time()
        print(f"--- /api/tts request completed successfully in {route_end_time - start_time_route:.2f} seconds ---")
        print("-" * 70)
        return jsonify({"audio_base64": audio_base64, "format": "wav"}), 200

    except Exception as tts_err:
        print(f"ERROR during TTS processing: {tts_err}")
        traceback.print_exc()
        error_type = type(tts_err).__name__
        error_details = f"Error during speech synthesis ({error_type})."
        if "out of memory" in str(tts_err).lower(): error_details += " Insufficient RAM/VRAM."
        elif "language" in str(tts_err).lower(): error_details += f" Language '{language}' might be unsupported."
        elif "speaker" in str(tts_err).lower(): error_details += f" Speaker '{selected_speaker}' might be invalid."
        elif "empty audio" in str(tts_err).lower(): error_details += " Model produced no audio output."
        return jsonify({"error": "Failed to generate speech.", "details": error_details}), 500


@app.route('/api/generate-course/<int:article_id>', methods=['POST'])
def generate_course_and_quiz(article_id):
    """
    Generates a course outline and quiz for a given article ID using Gemini
    and saves them to the database. Checks if they already exist first.
    """
    start_time_route = time.time()
    print(f"\n--- Received request for POST /api/generate-course/{article_id} ---")

    # --- 1. Check Agent Availability ---
    # (Keep this check first, no point proceeding if agent is down)
    if not course_quiz_agent or not course_quiz_agent.is_available:
        print("Error: Course/Quiz Generation Agent is not available (check API key and initialization logs).")
        return jsonify({"error": "Course and Quiz Generation service is currently unavailable."}), 503

    # --- 2. Validate Article ID ---
    if article_id <= 0:
        return jsonify({"error": "Invalid Article ID."}), 400

    conn = None
    cursor = None
    full_article_text = ""
    article_title = f"Article {article_id}" # Default title

    try:
        # --- 3. Fetch Article Details (including potential existing course/quiz) ---
        print(f"Fetching details for article ID: {article_id}")
        conn = get_db_connection()
        if not conn:
            print("Error: Failed to connect to database to fetch article details.")
            return jsonify({"error": "Database connection failed."}), 503

        cursor = conn.cursor(dictionary=True)
        # Fetch title, sections, insights, AND existing generated content
        query = """
            SELECT
                title, sections, Insights,
                generated_course, generated_quiz
            FROM articles
            WHERE id = %s
        """
        cursor.execute(query, (article_id,))
        article_data = cursor.fetchone()

        if not article_data:
            print(f"Error: Article with ID {article_id} not found in the database.")
            # Ensure connection is closed if we exit here
            if cursor: cursor.close()
            if conn and conn.is_connected(): conn.close()
            return jsonify({"error": f"Article with ID {article_id} not found."}), 404

        article_title = article_data.get('title', f"Article {article_id}")
        print(f"Found article: '{article_title[:100]}...'")

        # --- 3.5 CHECK IF COURSE AND QUIZ ALREADY EXIST ---
        existing_course = article_data.get('generated_course')
        existing_quiz = article_data.get('generated_quiz')

        # Check if both fields have substantial content (not None, not empty string)
        # Modify this check if your definition of "exists" is different (e.g., JSON validity)
        if (isinstance(existing_course, str) and existing_course.strip()) and \
           (isinstance(existing_quiz, str) and existing_quiz.strip()):
            print(f"Found existing course and quiz for article ID {article_id}. Skipping generation.")
            # Ensure connection is closed before returning
            if cursor: cursor.close()
            if conn and conn.is_connected(): conn.close()
            route_end_time = time.time()
            print(f"--- /api/generate-course/{article_id} request completed early (already exists) in {route_end_time - start_time_route:.2f} seconds ---")
            # Return a specific message and potentially the existing data
            return jsonify({
                "message": f"Course outline and quiz already exist for article ID {article_id}.",
                "article_id": article_id,
                "status": "already_exists",
                # Optionally return the existing data if the frontend might need it
                "generated_course": existing_course,
                "generated_quiz": existing_quiz
            }), 200 # 200 OK is appropriate, as the request is fulfilled (data exists)
        else:
             print(f"Existing course/quiz not found or incomplete for article ID {article_id}. Proceeding with generation.")
             # One or both are missing, continue with generation...

        # --- IF WE REACH HERE, GENERATION IS NEEDED ---

        # --- 4. Concatenate Text for Agent (Only if needed) ---
        # Start with title and insights (if available and not an error message)
        full_article_text += f"Title: {article_title}\n\n"
        insights = article_data.get('Insights')
        if insights and not any(err in insights.lower() for err in ["failed", "error", "unavailable", "not generated"]):
            full_article_text += f"Overall Insights:\n{insights}\n\n---\n\n"

        # Add text from sections
        sections_json = article_data.get('sections')
        sections_data = {}
        if sections_json:
            try:
                sections_data = json.loads(sections_json)
                if not isinstance(sections_data, dict):
                     print(f"Warning: Parsed sections JSON for article {article_id} is not a dictionary. Type: {type(sections_data)}")
                     sections_data = {} # Reset to empty dict if not a dict
            except json.JSONDecodeError:
                print(f"Warning: Could not parse sections JSON for article {article_id}. Proceeding without section text.")

        if isinstance(sections_data, dict) and sections_data:
            print(f"Extracting text from {len(sections_data)} sections...")
            section_count = 0
            for section_heading, section_content in sections_data.items():
                full_text = ""
                # Check if section_content is a dict and has 'full_text'
                if isinstance(section_content, dict) and isinstance(section_content.get("full_text"), str):
                    full_text = section_content["full_text"].strip()

                if full_text:
                    # Add section header and text
                    full_article_text += f"Section: {section_heading}\n\n{full_text}\n\n---\n\n"
                    section_count += 1
            print(f"Added text from {section_count} sections.")
        elif isinstance(sections_data, dict) and not sections_data:
             print(f"Info: Sections data for article {article_id} is an empty dictionary.")

        # Final check if any text was gathered
        full_article_text = full_article_text.strip() # Remove any trailing separators/whitespace
        if not full_article_text or len(full_article_text) < 100: # Arbitrary minimum length check
            print(f"Error: No substantial text content could be extracted from article {article_id} for generation. Text length: {len(full_article_text)}")
            # Ensure connection is closed if we exit here
            if cursor: cursor.close()
            if conn and conn.is_connected(): conn.close()
            return jsonify({"error": "Could not retrieve sufficient text content from the article to generate course/quiz."}), 400

        print(f"Total text length for generation: {len(full_article_text)} characters.")

        # --- 5. Call the Course/Quiz Agent ---
        # (Close the fetch cursor/connection BEFORE the potentially long API call)
        if cursor: cursor.close(); cursor = None
        if conn and conn.is_connected(): conn.close(); conn = None
        print("Closed initial DB connection before calling agent.")

        generation_result = course_quiz_agent.generate_course_and_quiz(full_article_text)

        if not generation_result or not isinstance(generation_result, dict):
            print("Error: Course/Quiz agent returned invalid data or failed.")
            details = "Agent returned unexpected data or None."
            if generation_result and isinstance(generation_result, str):
                 details = generation_result
            return jsonify({"error": "Failed to generate course and quiz content.", "details": details}), 500

        generated_course = generation_result.get("course")
        generated_quiz = generation_result.get("quiz")

        # Check if generation specifically failed
        course_failed = not generated_course or "Error:" in generated_course
        quiz_failed = not generated_quiz or "Error:" in generated_quiz

        if course_failed:
            error_detail = generated_course or "Course generation failed (no content returned)."
            print(f"Error from agent (Course): {error_detail}")
            return jsonify({"error": "Failed to generate course content.", "details": error_detail}), 500
        if quiz_failed:
            error_detail = generated_quiz or "Quiz generation failed (no content returned)."
            print(f"Error from agent (Quiz): {error_detail}")
            print(f"Note: Course generation succeeded, but quiz generation failed.")
            return jsonify({"error": "Failed to generate quiz content.", "details": error_detail}), 500

        print("Successfully generated course and quiz content via agent.")

        # --- 6. Save Generated Content to Database ---
        print(f"Saving generated course and quiz to database for article ID: {article_id}")
        # Connection should be closed, re-establish for the update
        conn = get_db_connection()
        if not conn:
            print("Error: Database connection failed while trying to save results.")
            return jsonify({"error": "Database connection failed while trying to save results."}), 503

        cursor = conn.cursor()
        update_query = """
            UPDATE articles
            SET generated_course = %s, generated_quiz = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """
        try:
            cursor.execute(update_query, (generated_course, generated_quiz, article_id))
            conn.commit()

            if cursor.rowcount == 0:
                print(f"Warning: Database update affected 0 rows for article ID {article_id}.")
                return jsonify({"error": "Failed to save generated content to database.", "details": "Article ID not found during update."}), 404

            print(f"Successfully saved course and quiz to database for article ID {article_id}.")

        except Error as db_update_err:
             print(f"Database Error during update for article {article_id}: {db_update_err}")
             if conn: conn.rollback()
             traceback.print_exc()
             return jsonify({"error": "Database error occurred during saving.", "details": str(db_update_err)}), 500
        except Exception as update_err:
             print(f"Unexpected Error during update for article {article_id}: {update_err}")
             if conn: conn.rollback()
             traceback.print_exc()
             return jsonify({"error": "An internal server error occurred during saving.", "details": str(update_err)}), 500
        finally:
            # Ensure update connection resources are closed
            if cursor: cursor.close()
            if conn and conn.is_connected(): conn.close()
            conn = None; cursor = None # Reset for clarity


        # --- 7. Return Success Response (after generation) ---
        route_end_time = time.time()
        print(f"--- /api/generate-course/{article_id} request completed in {route_end_time - start_time_route:.2f} seconds ---")
        return jsonify({
            "message": f"Successfully generated and saved course outline and quiz for article ID {article_id}.",
            "article_id": article_id,
            "status": "generated",
            "generated_course": generated_course, # Optionally return new data
            "generated_quiz": generated_quiz
        }), 200 # Use 200 OK for successful creation/update

    except Error as db_err: # Catch errors from initial fetch
        print(f"Database Error in /api/generate-course/{article_id} (initial fetch phase): {db_err}")
        # conn/cursor might be None or open here, finally block will handle
        traceback.print_exc()
        return jsonify({"error": "Database error occurred.", "details": str(db_err)}), 500
    except Exception as e: # Catch other unexpected errors
        print(f"Unexpected Error in /api/generate-course/{article_id}: {e}")
        # conn/cursor might be None or open here, finally block will handle
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500
    finally:
        # --- Robust Connection Closure ---
        # This block ensures resources are released regardless of where an error occurred
        # or if an early return happened (like finding existing data).
        print("Executing finally block for resource cleanup...")
        if cursor:
             try:
                 if not cursor.is_closed(): # Check if already closed
                     cursor.close()
                     print("Closed cursor in finally block.")
                 else:
                     print("Cursor already closed before finally block.")
             except Exception as cursor_close_err:
                 print(f"Error closing cursor in finally block: {cursor_close_err}")
        if conn and conn.is_connected():
             try:
                 conn.close()
                 print("Closed DB connection in finally block.")
             except Exception as conn_close_err:
                 print(f"Error closing connection in finally block: {conn_close_err}")
        elif conn and not conn.is_connected():
             print("DB connection already closed before finally block.")
        else:
             print("No active DB connection to close in finally block.")
        print("-" * 70)


# --- User Management Routes ---
# (Keep all existing user routes: /api/users, /api/login, etc.)
@app.route('/api/users', methods=['POST'])
def create_user():
    """Creates a new user."""
    print("\n--- Received request for POST /api/users ---")
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    if not username or not isinstance(username, str) or len(username) < 3: return jsonify({"error": "Invalid or missing username (min 3 chars)."}), 400
    if not email or not isinstance(email, str) or '@' not in email: return jsonify({"error": "Invalid or missing email."}), 400
    if not password or not isinstance(password, str) or len(password) < 6: return jsonify({"error": "Invalid or missing password (min 6 chars)."}), 400
    hashed_password = generate_password_hash(password)
    initial_recent = json.dumps([])
    conn = None; cursor = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "Database connection failed."}), 503
        cursor = conn.cursor()
        query = "INSERT INTO users (username, email, password_hash, recently_analyzed_ids) VALUES (%s, %s, %s, %s)"
        cursor.execute(query, (username, email, hashed_password, initial_recent))
        conn.commit()
        user_id = cursor.lastrowid
        print(f"User '{username}' created successfully with ID: {user_id}")
        return jsonify({"message": f"User '{username}' created successfully.", "user_id": user_id}), 201
    except IntegrityError as e:
        if conn: conn.rollback()
        error_msg = str(e).lower(); detail = "Database integrity error."
        if "duplicate entry" in error_msg:
             if "'username'" in error_msg: detail = f"Username '{username}' already exists."
             elif "'email'" in error_msg: detail = f"Email '{email}' already exists."
             else: detail = "Duplicate entry error."
             print(f"User creation failed: {detail}")
             return jsonify({"error": "User registration failed.", "details": detail}), 409
        else:
             print(f"User creation failed due to IntegrityError: {e}")
             return jsonify({"error": "Database error during user creation.", "details": detail}), 500
    except Error as e:
        if conn: conn.rollback()
        print(f"Database error creating user '{username}': {e}")
        return jsonify({"error": "Database error.", "details": str(e)}), 500
    except Exception as e:
        if conn: conn.rollback()
        print(f"Unexpected error creating user '{username}': {e}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error."}), 500
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()
        print("-" * 70)

@app.route('/api/users/<string:username>', methods=['GET'])
def get_user(username):
    """Gets details for a specific user (excluding password)."""
    print(f"\n--- Received request for GET /api/users/{username} ---")
    conn = None; cursor = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "Database connection failed."}), 503
        cursor = conn.cursor(dictionary=True)
        query = "SELECT id, username, email, recently_analyzed_ids, created_at, updated_at FROM users WHERE username = %s"
        cursor.execute(query, (username,))
        user_data = cursor.fetchone()
        if user_data:
            recent_json = user_data.get('recently_analyzed_ids')
            try: user_data['recently_analyzed_ids'] = json.loads(recent_json) if recent_json else []
            except (json.JSONDecodeError, TypeError):
                 print(f"Warning: Could not decode recently_analyzed_ids JSON for user {username}. Value: {recent_json}")
                 user_data['recently_analyzed_ids'] = []
            print(f"User '{username}' found.")
            return jsonify(user_data), 200
        else:
            print(f"User '{username}' not found.")
            return jsonify({"error": f"User '{username}' not found."}), 404
    except Error as e:
        print(f"Database error fetching user '{username}': {e}")
        return jsonify({"error": "Database error.", "details": str(e)}), 500
    except Exception as e:
        print(f"Unexpected error fetching user '{username}': {e}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error."}), 500
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()
        print("-" * 70)

@app.route('/api/users/<string:username>', methods=['PUT'])
def update_user(username):
    """Updates a user's email or password."""
    print(f"\n--- Received request for PUT /api/users/{username} ---")
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json(); new_email = data.get('email'); new_password = data.get('password')
    if not new_email and not new_password: return jsonify({"error": "No fields provided for update (email or password)."}), 400
    if new_email and (not isinstance(new_email, str) or '@' not in new_email): return jsonify({"error": "Invalid email format."}), 400
    if new_password and (not isinstance(new_password, str) or len(new_password) < 6): return jsonify({"error": "Invalid password (min 6 chars)."}), 400
    conn = None; cursor = None
    try:
        conn = get_db_connection()
        if not conn: return jsonify({"error": "Database connection failed."}), 503
        updates = []; params = []
        if new_email: updates.append("email = %s"); params.append(new_email)
        if new_password: updates.append("password_hash = %s"); params.append(generate_password_hash(new_password))
        if not updates: return jsonify({"error": "No valid fields to update."}), 400 # Should be redundant
        params.append(username)
        query = f"UPDATE users SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP WHERE username = %s"
        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        if cursor.rowcount == 0:
            conn.rollback(); print(f"Update failed: User '{username}' not found.")
            return jsonify({"error": f"User '{username}' not found."}), 404
        conn.commit(); print(f"User '{username}' updated successfully.")
        return jsonify({"message": f"User '{username}' updated successfully."}), 200
    except IntegrityError as e:
        if conn: conn.rollback(); error_msg = str(e).lower()
        if "duplicate entry" in error_msg and "'email'" in error_msg:
             detail = f"Email '{new_email}' is already in use."
             print(f"User update failed: {detail}")
             return jsonify({"error": "Update failed.", "details": detail}), 409
        else:
             print(f"User update failed due to IntegrityError: {e}")
             return jsonify({"error": "Database error during user update.", "details": "Integrity constraint violated."}), 500
    except Error as e:
        if conn: conn.rollback(); print(f"Database error updating user '{username}': {e}")
        return jsonify({"error": "Database error.", "details": str(e)}), 500
    except Exception as e:
        if conn: conn.rollback(); print(f"Unexpected error updating user '{username}': {e}")
        traceback.print_exc(); return jsonify({"error": "Internal server error."}), 500
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()
        print("-" * 70)

MAX_RECENT_ANALYSES_DB = 10

@app.route('/api/users/<string:username>/recent_analyses', methods=['PATCH'])
def add_recent_analysis_id(username):
    """Adds a new unique analysis ID to the user's recent list."""
    print(f"\n--- Received request for PATCH /api/users/{username}/recent_analyses ---")
    if not request.is_json: print("Request error: Payload must be JSON."); return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json(); analysis_id_raw = data.get('analysis_id')
    if analysis_id_raw is None: print("Request error: 'analysis_id' missing."); return jsonify({"error": "'analysis_id' is required."}), 400
    try: analysis_id = int(analysis_id_raw)
    except (ValueError, TypeError):
         print(f"Request error: 'analysis_id' must be an integer. Received: {analysis_id_raw} (type: {type(analysis_id_raw)})")
         return jsonify({"error": "'analysis_id' must be an integer."}), 400
    conn = None; cursor = None
    try:
        conn = get_db_connection();
        if not conn: print("Error: Failed to establish database connection."); return jsonify({"error": "Database connection failed."}), 503
        cursor = conn.cursor(dictionary=True)
        print(f"Fetching current user data for user '{username}'...");
        cursor.execute("SELECT recently_analyzed_ids FROM users WHERE username = %s FOR UPDATE", (username,))
        user_data = cursor.fetchone()
        if not user_data: print(f"Update failed: User '{username}' not found."); return jsonify({"error": f"User '{username}' not found."}), 404
        current_ids_raw = user_data.get('recently_analyzed_ids'); current_ids = []
        if isinstance(current_ids_raw, str):
            try:
                parsed_list = json.loads(current_ids_raw)
                if isinstance(parsed_list, list): current_ids = parsed_list
                else: print(f"DB field 'recently_analyzed_ids' for '{username}' contained valid JSON, but not a list. Resetting.")
            except json.JSONDecodeError: print(f"DB field 'recently_analyzed_ids' for '{username}' contained invalid JSON string. Resetting.")
        elif isinstance(current_ids_raw, list): current_ids = current_ids_raw
        elif current_ids_raw is None: print(f"Initializing empty list for '{username}' (DB value was NULL).")
        else: print(f"Initializing/resetting list for '{username}'. Received unexpected type: {type(current_ids_raw)}.")
        print(f"Current IDs (processed) for '{username}': {current_ids}")
        id_changed = False; cleaned_ids = []; needs_cleaning = False
        for item in current_ids:
            try: cleaned_ids.append(int(item))
            except (ValueError, TypeError): needs_cleaning = True; print(f"  - Warning: Found non-integer item '{item}' for '{username}'. Skipping.")
        if needs_cleaning: print(f"Original list for '{username}' cleaned: {current_ids}"); current_ids = cleaned_ids; id_changed = True
        if analysis_id not in current_ids:
            current_ids.insert(0, analysis_id); print(f"Added new ID {analysis_id}."); id_changed = True
            if len(current_ids) > MAX_RECENT_ANALYSES_DB: removed_id = current_ids.pop(); print(f"List exceeded max ({MAX_RECENT_ANALYSES_DB}). Removed: {removed_id}")
        else: print(f"ID {analysis_id} already exists.")
        print(f"Final IDs for '{username}': {current_ids}")
        if id_changed:
            final_ids_to_save = [int(i) for i in current_ids]; updated_ids_json_string = json.dumps(final_ids_to_save)
            print(f"Updating DB for '{username}' with: {updated_ids_json_string}")
            cursor.close(); cursor = conn.cursor() # Use non-dict cursor for update
            cursor.execute("UPDATE users SET recently_analyzed_ids = %s, updated_at = CURRENT_TIMESTAMP WHERE username = %s", (updated_ids_json_string, username))
            if cursor.rowcount == 0:
                conn.rollback(); print(f"Update failed unexpectedly for user '{username}' (rowcount 0).");
                return jsonify({"error": f"Failed to update user '{username}', user may no longer exist."}), 404
            conn.commit(); print(f"Successfully updated recently_analyzed_ids for user '{username}'.")
            return jsonify({"message": f"Updated recent analyses for user '{username}'.", "recently_analyzed_ids": final_ids_to_save}), 200
        else:
             print(f"No database update needed for user '{username}'.")
             final_ids_display = [int(i) for i in current_ids]
             return jsonify({"message": f"Analysis ID {analysis_id} already in recent list for user '{username}'.", "recently_analyzed_ids": final_ids_display}), 200
    except Error as e:
        if conn: conn.rollback(); print(f"Database error updating recent analyses for '{username}': {e}"); traceback.print_exc()
        return jsonify({"error": "Database error.", "details": str(e)}), 500
    except Exception as e:
         if conn: conn.rollback(); print(f"Unexpected error updating recent analyses for '{username}': {e}"); traceback.print_exc()
         return jsonify({"error": "Internal server error."}), 500
    finally:
        if cursor:
            try:
                cursor.close()
            except Exception as e_cur:
                print(f"Error closing cursor: {e_cur}")
        if conn and conn.is_connected():
            try:
                conn.close()
                print("Database connection closed.")
            except Exception as e_con:
                print(f"Error closing connection: {e_con}")
        print("-" * 70)

@app.route('/api/users/<string:username>', methods=['DELETE'])
def delete_user(username):
    """Deletes a user."""
    print(f"\n--- Received request for DELETE /api/users/{username} ---")
    conn = None; cursor = None
    try:
        conn = get_db_connection();
        if not conn: return jsonify({"error": "Database connection failed."}), 503
        cursor = conn.cursor()
        query = "DELETE FROM users WHERE username = %s"
        cursor.execute(query, (username,))
        if cursor.rowcount == 0: print(f"Deletion failed: User '{username}' not found."); return jsonify({"error": f"User '{username}' not found."}), 404
        conn.commit(); print(f"User '{username}' deleted successfully.")
        return '', 204
    except Error as e:
        if conn: conn.rollback(); print(f"Database error deleting user '{username}': {e}")
        return jsonify({"error": "Database error.", "details": str(e)}), 500
    except Exception as e:
        if conn: conn.rollback(); print(f"Unexpected error deleting user '{username}': {e}"); traceback.print_exc()
        return jsonify({"error": "Internal server error."}), 500
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()
        print("-" * 70)

@app.route('/api/login', methods=['POST'])
def login_user():
    """Authenticates a user."""
    print("\n--- Received request for POST /api/login ---")
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json(); username = data.get('username'); password = data.get('password')
    if not username or not password: return jsonify({"error": "Missing username or password"}), 400
    conn = None; cursor = None
    try:
        conn = get_db_connection();
        if not conn: return jsonify({"error": "Database connection failed."}), 503
        cursor = conn.cursor(dictionary=True)
        query = "SELECT id, username, email, password_hash, recently_analyzed_ids FROM users WHERE username = %s"
        cursor.execute(query, (username,))
        user = cursor.fetchone()
        if user and check_password_hash(user['password_hash'], password):
            print(f"User '{username}' authenticated successfully.")
            recent_json = user.get('recently_analyzed_ids')
            try: recent_list = json.loads(recent_json) if recent_json else []
            except (json.JSONDecodeError, TypeError): print(f"Warning: Could not decode recently_analyzed_ids JSON during login for user {username}."); recent_list = []
            user_info = {"id": user['id'], "username": user['username'], "email": user['email'], "recently_analyzed_ids": recent_list}
            return jsonify(user_info), 200
        else:
            print(f"Authentication failed for user '{username}'.")
            return jsonify({"error": "Invalid username or password"}), 401
    except Error as e:
        print(f"Database error during login for '{username}': {e}")
        return jsonify({"error": "Database error during login.", "details": str(e)}), 500
    except Exception as e:
        print(f"Unexpected error during login for '{username}': {e}"); traceback.print_exc()
        return jsonify({"error": "Internal server error during login."}), 500
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()
        print("-" * 70)

# --- Batch Article Details Route ---
@app.route('/api/articles/details', methods=['GET'])
def get_article_details_batch():
    """Fetches details for multiple articles based on IDs."""
    print("\n--- Received request for GET /api/articles/details ---")
    id_string = request.args.get('ids')
    if not id_string: return jsonify({"error": "Missing 'ids' query parameter"}), 400
    try:
        article_ids = [int(id_str) for id_str in id_string.split(',') if id_str.strip().isdigit() and int(id_str) > 0]
        if not article_ids: raise ValueError("No valid positive integer IDs found.")
    except ValueError as e:
        print(f"Error parsing article IDs: {e}")
        return jsonify({"error": "Invalid format for 'ids'. Must be comma-separated positive integers."}), 400
    conn = None; cursor = None; articles_dict = {}
    try:
        conn = get_db_connection();
        if not conn: return jsonify({"error": "Database connection failed"}), 503
        cursor = conn.cursor(dictionary=True)
        placeholders = ','.join(['%s'] * len(article_ids))
        # Fetch fields needed for card display (e.g., id, title, insights, pages)
        # Can add generated_course/quiz status if needed (e.g., SELECT id, title, Insights, Pages, (generated_course IS NOT NULL) AS has_course FROM ...)
        query = f"SELECT id, title, Insights, Pages FROM articles WHERE id IN ({placeholders})"
        cursor.execute(query, tuple(article_ids))
        results = cursor.fetchall()
        for row in results:
            article_id = row['id']; description = "No description available."
            insights_text = row.get('Insights')
            if insights_text and not any(err in insights_text.lower() for err in ["failed", "error", "unavailable", "not generated"]):
                 description = insights_text[:250] + ('...' if len(insights_text) > 250 else '')
            # Removed fallback to sections for brevity, insights should be the primary source now
            articles_dict[article_id] = {
                "id": article_id, "title": row.get('title', 'Unknown Title'),
                "description": description, "pages": row.get('Pages')
            }
        ordered_articles = [articles_dict[req_id] for req_id in article_ids if req_id in articles_dict]
        print(f"Returning details for {len(ordered_articles)} articles.")
        return jsonify(ordered_articles), 200
    except Error as e:
        print(f"Database error fetching article details: {e}")
        return jsonify({"error": "Database error fetching article details"}), 500
    except Exception as e:
        print(f"Unexpected error fetching article details: {e}"); traceback.print_exc()
        return jsonify({"error": "Internal server error fetching article details"}), 500
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()
        print("-" * 70)


# --- NEW ENDPOINT 1: Identify Themes from Multiple Articles ---
@app.route('/api/multi-article/identify-themes', methods=['POST'])
def identify_multi_article_themes():
    start_time_route = time.time()
    print("\n--- Received POST /api/multi-article/identify-themes ---")

    if not advanced_course_agent or not advanced_course_agent.is_available:
        return jsonify({"error": "Advanced Course Generation service unavailable."}), 503
    if not extraction_agent:
        return jsonify({"error": "Extraction service unavailable."}), 503

    if 'pdf_files' not in request.files:
        return jsonify({"error": "No 'pdf_files' part in the request. Please send files under this key."}), 400
    
    pdf_files_list = request.files.getlist('pdf_files')
    if not pdf_files_list or not any(f.filename for f in pdf_files_list):
        return jsonify({"error": "No PDF files selected or files are empty."}), 400

    extracted_articles_content = []
    article_texts_for_agent = []
    temp_files_to_clean = []

    output_dir = app.config.get('OUTPUT_DIR')
    if not output_dir:
        return jsonify({"error": "Server configuration error (output path)."}), 500

    try:
        for i, pdf_file in enumerate(pdf_files_list):
            if not pdf_file.filename or not pdf_file.filename.lower().endswith('.pdf'):
                print(f"Skipping non-PDF file or file with no name: {pdf_file.filename}")
                continue

            safe_basename = "".join(c if c.isalnum() or c in ['_', '.', '-'] else '_' for c in ntpath.basename(pdf_file.filename))
            temp_pdf_path = os.path.join(output_dir, f"multi_upload_{int(time.time())}_{i}_{safe_basename}")
            temp_files_to_clean.append(temp_pdf_path)
            
            try:
                pdf_file.save(temp_pdf_path)
                print(f"Saved temporary PDF for multi-article processing: {temp_pdf_path}")

                extraction_result = extraction_agent.extract(temp_pdf_path)
                if not extraction_result or not extraction_result.get("title"):
                    print(f"Warning: Failed to extract meaningful data from {pdf_file.filename}")
                    # Optionally add a placeholder or skip
                    extracted_articles_content.append({
                        "original_filename": pdf_file.filename,
                        "extracted_text": f"Error: Could not extract text from {pdf_file.filename}",
                        "error": "Extraction failed"
                    })
                    continue # Skip this article for theme identification if extraction fails severely

                # Construct full text for this article
                current_article_text = f"Title: {extraction_result.get('title', 'No Title Found')}\n\n"
                abstract = extraction_result.get('abstract')
                if abstract and isinstance(abstract, str):
                    current_article_text += f"Abstract:\n{abstract}\n\n---\n\n"
                
                sections = extraction_result.get('sections', {})
                if isinstance(sections, dict):
                    for sec_title, sec_content in sections.items():
                        full_text = sec_content.get('full_text') if isinstance(sec_content, dict) else None
                        if full_text and isinstance(full_text, str):
                            current_article_text += f"Section: {sec_title}\n\n{full_text.strip()}\n\n---\n\n"
                
                current_article_text = current_article_text.strip()
                if len(current_article_text) < 50: # Basic check
                    print(f"Warning: Extracted text for {pdf_file.filename} seems too short. Content: {current_article_text[:100]}...")
                    # Decide if to include or not
                
                article_texts_for_agent.append(current_article_text)
                extracted_articles_content.append({
                    "original_filename": pdf_file.filename,
                    "extracted_text": current_article_text
                })

            except Exception as e_extract:
                print(f"Error processing file {pdf_file.filename}: {e_extract}")
                traceback.print_exc()
                extracted_articles_content.append({
                    "original_filename": pdf_file.filename,
                    "extracted_text": f"Error during extraction: {str(e_extract)}",
                    "error": str(e_extract)
                })
                # Continue to next file if one fails
            finally:
                # Clean up individual temp file immediately after processing or if error
                if os.path.exists(temp_pdf_path) and temp_pdf_path in temp_files_to_clean:
                    try:
                        os.remove(temp_pdf_path)
                        temp_files_to_clean.remove(temp_pdf_path) # remove from list if deleted
                        print(f"Cleaned up temp file: {temp_pdf_path}")
                    except OSError as e_del:
                        print(f"Error deleting temp file {temp_pdf_path}: {e_del}")
        
        if not article_texts_for_agent:
            return jsonify({"error": "No articles could be successfully processed for text extraction."}), 400

        print(f"Identifying themes from {len(article_texts_for_agent)} processed articles...")
        identified_themes = advanced_course_agent.identify_course_themes_from_articles(article_texts_for_agent)

        if not identified_themes or "Error:" in identified_themes:
            details = identified_themes or "Theme identification failed (no content returned by agent)."
            return jsonify({"error": "Failed to identify course themes.", "details": details}), 500

        route_end_time = time.time()
        processing_time = route_end_time - start_time_route
        print(f"--- /api/multi-article/identify-themes completed in {processing_time:.2f}s ---")
        
        return jsonify({
            "message": "Themes identified successfully.",
            "identified_themes": identified_themes,
            "extracted_articles_content": extracted_articles_content # Send back the texts for next step
        }), 200

    except Exception as e:
        print(f"Unexpected error in /api/multi-article/identify-themes: {e}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error during theme identification.", "details": str(e)}), 500
    finally:
        # Clean up any remaining temp files (e.g., if main loop broke)
        for temp_file in temp_files_to_clean:
            if os.path.exists(temp_file):
                try: os.remove(temp_file); print(f"Final cleanup of temp file: {temp_file}")
                except OSError as e_del: print(f"Error in final cleanup of {temp_file}: {e_del}")
        print("-" * 70)


# --- NEW ENDPOINT 2: Generate Course from Selected Themes and Multiple Articles ---
@app.route('/api/multi-article/generate-course', methods=['POST'])
def generate_multi_article_course_from_themes():
    start_time_route = time.time()
    print("\n--- Received POST /api/multi-article/generate-course ---")

    if not advanced_course_agent or not advanced_course_agent.is_available:
        return jsonify({"error": "Advanced Course Generation service unavailable."}), 503
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON."}), 400

    data = request.get_json()
    article_contents_payload = data.get('article_contents') # List of {"original_filename": ..., "extracted_text": ...}
    selected_themes_text = data.get('selected_themes_text')

    if not isinstance(article_contents_payload, list) or not article_contents_payload:
        return jsonify({"error": "Missing or invalid 'article_contents'. Must be a non-empty list."}), 400
    if not selected_themes_text or not isinstance(selected_themes_text, str):
        return jsonify({"error": "Missing or invalid 'selected_themes_text'."}), 400

    article_texts_for_agent = []
    for item in article_contents_payload:
        if isinstance(item, dict) and isinstance(item.get("extracted_text"), str) and item["extracted_text"].strip():
            article_texts_for_agent.append(item["extracted_text"])
        else:
            # Log a warning or decide how to handle malformed items
            print(f"Warning: Skipping an item in 'article_contents' due to missing or invalid 'extracted_text'. Item: {item.get('original_filename', 'Unknown')}")
    
    if not article_texts_for_agent:
         return jsonify({"error": "No valid article texts found in 'article_contents' payload."}), 400

    print(f"Generating course from {len(article_texts_for_agent)} articles and selected themes...")
    
    try:
        generated_course_outline = advanced_course_agent.generate_course_from_themes(
            article_texts_for_agent,
            selected_themes_text
        )

        if not generated_course_outline or "Error:" in generated_course_outline:
            details = generated_course_outline or "Course generation from themes failed (no content by agent)."
            return jsonify({"error": "Failed to generate course from themes.", "details": details}), 500
        
        # Optionally, generate a quiz for this multi-article course outline
        # generated_quiz_for_multi = advanced_course_agent.generate_quiz_from_outline(generated_course_outline)
        # if not generated_quiz_for_multi or "Error:" in generated_quiz_for_multi:
        #     print(f"Warning: Quiz generation failed for multi-article course. Details: {generated_quiz_for_multi}")
        #     generated_quiz_for_multi = None # Or the error string

        route_end_time = time.time()
        processing_time = route_end_time - start_time_route
        print(f"--- /api/multi-article/generate-course completed in {processing_time:.2f}s ---")

        response_data = {
            "message": "Course generated successfully from selected themes and articles.",
            "generated_course_outline": generated_course_outline,
            # "generated_quiz": generated_quiz_for_multi # If you decide to include it
        }
        # This course is NOT saved to the database by this endpoint.
        return jsonify(response_data), 200

    except Exception as e:
        print(f"Unexpected error in /api/multi-article/generate-course: {e}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error during course generation.", "details": str(e)}), 500
    finally:
        print("-" * 70)


# --- Single Article Details Route ---
@app.route('/api/articles/<int:article_id>', methods=['GET'])
def get_article_details(article_id):
    """Fetches all details for a single article including generated course/quiz."""
    print(f"\n--- Received request for GET /api/articles/{article_id} ---")
    if article_id <= 0:
        print(f"Error: Invalid article ID requested: {article_id}")
        return jsonify({"error": "Article ID must be a positive integer."}), 400
    conn = None; cursor = None
    try:
        conn = get_db_connection()
        if not conn: print("Error: Database connection failed"); return jsonify({"error": "Database connection failed"}), 503
        cursor = conn.cursor(dictionary=True)
        # Select ALL columns including the new ones
        query = """
            SELECT
                id, title, sections, Insights, figures, tables,
                created_at, updated_at, Pages,
                generated_course, generated_quiz
            FROM articles WHERE id = %s
        """
        cursor.execute(query, (article_id,))
        article = cursor.fetchone()
        if article:
            if article.get('created_at'): article['created_at'] = article['created_at'].isoformat()
            if article.get('updated_at'): article['updated_at'] = article['updated_at'].isoformat()
            for field in ['sections', 'figures', 'tables']:
                if article.get(field) and isinstance(article[field], str):
                    try: article[field] = json.loads(article[field])
                    except json.JSONDecodeError: print(f"Warning: Could not parse JSON for field '{field}' in article {article_id}.")
            print(f"Successfully retrieved article ID: {article_id} (including course/quiz fields)")
            return jsonify(article), 200
        else:
            print(f"Error: Article not found for ID: {article_id}")
            return jsonify({"error": f"Article with ID {article_id} not found"}), 404
    except Error as e:
        print(f"Database error fetching article {article_id}: {e}")
        return jsonify({"error": "Database error while fetching article details"}), 500
    except Exception as e:
        print(f"Unexpected error fetching article {article_id}: {e}"); traceback.print_exc()
        return jsonify({"error": "Internal server error while fetching article details"}), 500
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()
        print("-" * 70)


# --- Update Section Summary by Heading ---
@app.route('/api/analyses/<int:analysis_id>/sections/summary', methods=['PATCH'])
def update_section_summary_by_heading(analysis_id):
    start_time_route = time.time()
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    new_summary = data.get('summary')
    section_heading_to_update = data.get('heading')

    print(f"\n--- Received request for PATCH /api/analyses/{analysis_id}/sections/summary ---")
    print(f"Body: analysis_id={analysis_id}, heading='{section_heading_to_update}', new_summary='{str(new_summary)[:50]}...'")

    if new_summary is None or not isinstance(new_summary, str): # Allow empty string
        return jsonify({"error": "Invalid or missing 'summary' field. Must be a string."}), 400
    if not section_heading_to_update or not isinstance(section_heading_to_update, str):
        return jsonify({"error": "Invalid or missing 'heading' field for the section to update."}), 400
    if analysis_id <= 0:
        return jsonify({"error": "Invalid 'analysis_id'."}), 400
    
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if not conn:
            print("Error: Database connection failed.")
            return jsonify({"error": "Database connection failed."}), 503

        cursor = conn.cursor(dictionary=True) 

        print(f"Fetching article {analysis_id} to update section '{section_heading_to_update}'...")
        query_select = "SELECT sections FROM articles WHERE id = %s" # FOR UPDATE can be added if high concurrency
        cursor.execute(query_select, (analysis_id,))
        article_data = cursor.fetchone()

        if not article_data:
            print(f"Error: Article with ID {analysis_id} not found.")
            return jsonify({"error": f"Article with ID {analysis_id} not found."}), 404

        sections_json_str = article_data.get('sections')
        try:
            sections_data_ordered = json.loads(sections_json_str, object_pairs_hook=OrderedDict) if sections_json_str else OrderedDict()
        except json.JSONDecodeError as json_err:
            print(f"Error: Could not parse 'sections' JSON for article {analysis_id}. Error: {json_err}. Content: {sections_json_str[:200]}")
            return jsonify({"error": "Failed to parse existing sections data from database."}), 500
        
        if section_heading_to_update not in sections_data_ordered:
            print(f"Error: Section with heading '{section_heading_to_update}' not found in article {analysis_id}.")
            # For debugging, list available headings if not found
            print(f"Available headings: {list(sections_data_ordered.keys())}")
            return jsonify({"error": f"Section with heading '{section_heading_to_update}' not found in article {analysis_id}."}), 404

        # Update the summary in the OrderedDict
        target_section_content = sections_data_ordered[section_heading_to_update]
        if isinstance(target_section_content, dict):
            target_section_content['summary'] = new_summary
        else:
            # This case means the section content was not a dictionary as expected.
            # This could happen if the initial data extraction stored it differently.
            # We'll re-structure it now.
            print(f"Warning: Content for section '{section_heading_to_update}' was not a dict (type: {type(target_section_content)}). Re-structuring.")
            sections_data_ordered[section_heading_to_update] = {
                "full_text": str(target_section_content), # Preserve old content as full_text
                "summary": new_summary
            }
            
        updated_sections_json = json.dumps(sections_data_ordered) # OrderedDict ensures order in JSON string

        print(f"Updating database for article {analysis_id} with new sections JSON.")
        query_update = "UPDATE articles SET sections = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s"
        
        # Close dictionary cursor and get a standard one for update if it was a dict cursor
        if hasattr(cursor, 'description') and cursor.description: 
            cursor.close()
            cursor = conn.cursor()

        cursor.execute(query_update, (updated_sections_json, analysis_id))
        conn.commit()

        if cursor.rowcount == 0:
            print(f"Warning: Database update affected 0 rows for article ID {analysis_id} (summary update).")
            return jsonify({"error": "Failed to save updated summary. Article may have been modified or deleted."}), 404
        
        print(f"Summary for section '{section_heading_to_update}' of article {analysis_id} updated successfully.")
        route_end_time = time.time()
        print(f"--- Update summary by heading request completed in {route_end_time - start_time_route:.2f} seconds ---")
        
        return jsonify({
            "message": "Summary updated successfully.",
            "analysis_id": analysis_id,
            "section_heading": section_heading_to_update, # Echo back for confirmation
            "updated_summary": new_summary
        }), 200

    except Error as db_err:
        print(f"Database Error during summary update for article {analysis_id}: {db_err}")
        if conn: conn.rollback()
        traceback.print_exc()
        return jsonify({"error": "Database error occurred during summary update.", "details": str(db_err)}), 500
    except json.JSONDecodeError as json_err:
        print(f"JSON Decode Error for article {analysis_id}: {json_err}")
        if conn: conn.rollback()
        traceback.print_exc()
        return jsonify({"error": "Error processing section data structure.", "details": str(json_err)}), 500
    except Exception as e:
        print(f"Unexpected Error during summary update for article {analysis_id}: {e}")
        if conn: conn.rollback()
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500
    finally:
        if cursor:
            try: cursor.close()
            except Exception: pass
        if conn and conn.is_connected():
            try: conn.close()
            except Exception: pass
        print("-" * 70)


# --- Run Flask App ---
if __name__ == "__main__":
    print("\n" + "="*30 + " Starting Flask Server " + "="*30)
    print("\n--- Final Agent & Service Status ---")
    # Extraction Agent
    print(f"Extraction Agent uses:     GROBID '{GROBID_API}', pdffigures '{os.path.basename(PDFFIGURES_JAR_PATH)}'")
    # Summarization Agent
    # summarization_status = 'Unavailable'
    # if summarization_agent and hasattr(summarization_agent, 'is_local_pipeline_available'):
    #     summarization_status = 'Available' if summarization_agent.is_local_pipeline_available else 'Unavailable (Pipeline Failed)'
    # print(f"Summarization Agent:       {summarization_status} (on '{summarizer_device_name}')")
    summarization_status_gemini = 'Unavailable'
    if summarization_agent and hasattr(summarization_agent, 'is_llm_available'):
        summarization_status_gemini = 'Available (Gemini)' if summarization_agent.is_llm_available else 'Unavailable (Gemini Not Configured)'
    print(f"Summarization Agent:       {summarization_status_gemini}")
    if summarization_agent and summarization_agent.is_llm_available:
        print(f"  > Gemini Model:          {summarization_agent.llm_model_name}")
    # QA Agent
    qa_ready_status = 'Unavailable'
    if qa_agent:
        qa_ready_status = 'Available' if qa_agent.is_ready else f'Unavailable ({qa_agent.get_missing_components_message()})'
    print(f"QA Agent:                  {qa_ready_status}")
    print(f"  > QA Pipeline (Local):   {qa_pipeline_device_name}") # Note if local QA is used
    print(f"  > Embedding Model:       {embedding_model_device_name}")
    print(f"  > Cross-Encoder Model:   {cross_encoder_device_name}")
    # Translation Agent
    print(f"Translation Agent:         {'Available' if translation_agent and translation_agent.is_available else 'Unavailable'}")
    # Course/Quiz Agent <<< ADDED STATUS
    print(f"Course/Quiz Agent (Gemini):{'Available' if course_quiz_agent and course_quiz_agent.is_available else 'Unavailable (Check API Key/Logs)'}")
    # TTS
    print(f"Text-to-Speech (TTS):      {tts_status}")
    if tts_instance and tts_speakers: print(f"  > Default TTS Speaker:   {tts_speakers[0] if tts_speakers else 'N/A'}")
    if tts_instance and tts_languages: print(f"  > Default TTS Language:  {tts_languages[0] if tts_languages else 'N/A'}")
    print("-" * 70)
    # ChromaDB
    print(f"ChromaDB Client Status:    {'Initialized' if chroma_client else 'Failed/Skipped'}")
    if chroma_collection:
        try:
             count = chroma_collection.count()
             print(f"ChromaDB Collection:       '{CHROMA_COLLECTION_NAME}' (Count: {count})")
             chroma_path_final = app.config.get('CHROMA_DB_PATH', 'Path Not Set')
             print(f"ChromaDB Path:             '{chroma_path_final}'")
        except Exception as e_count:
            print(f"ChromaDB Collection:       '{CHROMA_COLLECTION_NAME}' (Error getting count: {e_count})")
    else:
        print(f"ChromaDB Collection:       Not Initialized")
    print("-" * 70)
    # Paths and DB
    output_dir_final = app.config.get('OUTPUT_DIR', 'Path Not Set')
    print(f"File Output Directory:     '{output_dir_final}'")
    db_host = DB_CONFIG.get('host', 'Not Set')
    db_name = DB_CONFIG.get('database', 'Not Set')
    print(f"SQL Database Target:       '{db_host}/{db_name}'")
    print("-" * 70)
    print(f"Flask App Ready. Listening on http://0.0.0.0:5000")
    print("="*90 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)