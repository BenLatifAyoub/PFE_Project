# # extraction_agent.py
# import os
# import requests
# from bs4 import BeautifulSoup
# import subprocess
# import json
# import time
# import shutil
# import glob
# import ntpath
# import traceback
# # import tempfile # No longer needed
# import base64
# import re
# from typing import List, Tuple, Optional, Dict, Any
# import pypdf # <--- ADDED: Import pypdf

# class ExtractionAgent:
#     def __init__(self, grobid_url, pdffigures_jar, pdffigures_output_dir="pdffigures2_output"):
#         self.grobid_url = grobid_url
#         self.pdffigures_jar = pdffigures_jar
#         self.pdffigures_output_dir = pdffigures_output_dir
#         if not os.path.exists(self.pdffigures_jar):
#             print(f"WARNING: pdffigures2 JAR not found at {self.pdffigures_jar}. Figure/Table extraction will fail.")
#         else:
#             print(f"ExtractionAgent: Found pdffigures2 JAR at {self.pdffigures_jar}")
#         print(f"ExtractionAgent: Using GROBID API at {self.grobid_url}")
#         print(f"ExtractionAgent: pdffigures2 output base directory: '{self.pdffigures_output_dir}'")

#     # --- NEW METHOD: Get page count directly from PDF ---
#     def _get_page_count_from_pdf(self, pdf_path: str) -> int:
#         """
#         Gets the page count directly from the PDF file using pypdf.
#         Returns 0 if the file cannot be read or is not a valid PDF.
#         """
#         print(f"ExtractionAgent: Reading page count from '{pdf_path}' using pypdf...")
#         try:
#             reader = pypdf.PdfReader(pdf_path)
#             count = len(reader.pages)
#             print(f"ExtractionAgent: pypdf reported {count} pages.")
#             return count
#         except pypdf.errors.PdfReadError as pdf_err:
#             print(f"Error: pypdf could not read PDF '{pdf_path}'. It might be corrupted or encrypted. Error: {pdf_err}")
#             return 0
#         except FileNotFoundError:
#             print(f"Error: PDF file not found at '{pdf_path}' for page count.")
#             return 0
#         except Exception as e:
#             print(f"Error: Unexpected error getting page count with pypdf: {e}")
#             traceback.print_exc()
#             return 0
#     # --- END NEW METHOD ---

#     def _get_grobid_xml(self, pdf_path):
#         print("ExtractionAgent: Requesting GROBID processing...")
#         try:
#             with open(pdf_path, "rb") as f:
#                 response = requests.post(
#                     self.grobid_url,
#                     files={"input": f},
#                     data={"consolidateHeader": "1", "consolidateCitations": "0"},
#                     timeout=300 # 5 minutes timeout
#                 )
#                 response.raise_for_status()
#                 print("ExtractionAgent: GROBID request successful.")
#                 return response.text
#         except requests.exceptions.Timeout:
#             print("Error: GROBID request timed out.")
#             return None
#         except requests.exceptions.RequestException as e:
#             if isinstance(e, requests.exceptions.ConnectionError):
#                 print(f"FATAL: GROBID connection failed at {self.grobid_url}. Is the GROBID server running? Error: {e}")
#             else:
#                 print(f"Error: GROBID connection/request failed: {e}")
#             return None
#         except Exception as e:
#             print(f"Error: Unexpected error during GROBID request: {e}")
#             traceback.print_exc()
#             return None

#     def _parse_grobid_xml(self, xml_content):
#         print("ExtractionAgent: Parsing GROBID XML for text structure...") # Updated log msg
#         if not xml_content:
#             print("Error: Cannot parse empty GROBID XML content.")
#             return None
#         try:
#             soup = BeautifulSoup(xml_content, "lxml-xml") # Use lxml-xml for better XML parsing

#             # Extract title
#             title_tag = soup.select_one("teiHeader > fileDesc > titleStmt > title")
#             title = title_tag.get_text(strip=True) if title_tag else "No title found"

#             # Extract abstract
#             abstract_tag = soup.select_one("teiHeader > profileDesc > abstract")
#             abstract = abstract_tag.get_text(strip=True) if abstract_tag else ""

#             # Initialize sections dictionary
#             sections = {}

#             # Add abstract to sections if present
#             if abstract:
#                 # Use a consistent key like 'Abstract' or '0. Abstract'
#                 sections["Abstract"] = {"full_text": abstract, "summary": None}

#             # Parse body sections
#             body = soup.select_one("text > body")
#             if body:
#                 top_level_keywords = [
#                     "abstract", "background", "introduction", "related work", "method",
#                     "methodology", "materials and methods", "experiment", "results",
#                     "discussion", "conclusion", "acknowledgement", "reference",
#                     "appendix", "supplementary"
#                 ]
#                 current_section_heading = "Introduction" # Default if first section has no heading
#                 section_counter = 1 # For unnamed sections

#                 for div in body.find_all("div", recursive=False):
#                     head = div.find("head")
#                     heading_text = "Unnamed Section"
#                     if head:
#                         heading_text = re.sub(r"^\s*(\d+\.|\w\.|[IVXLCDM]+\.)\s*", "", head.get_text(strip=True)).strip()
#                         if not heading_text:
#                              heading_text = f"Section {head.get('n', section_counter)}"
#                              section_counter += 1
#                     else:
#                         heading_text = f"Unnamed Section {section_counter}"
#                         section_counter += 1

#                     is_major_section = any(keyword in heading_text.lower() for keyword in top_level_keywords)

#                     paragraphs = div.find_all("p", recursive=False)
#                     div_text_nodes = div.find_all(string=True, recursive=False)
#                     direct_text = "\n".join(t.strip() for t in div_text_nodes if t.strip() and not t.parent.name in ['head', 'figure', 'table'])
#                     para_text = "\n".join(p.get_text(strip=True) for p in paragraphs)
#                     full_section_text = (para_text + "\n" + direct_text).strip()

#                     if not full_section_text:
#                          continue

#                     if is_major_section:
#                         current_section_heading = heading_text
#                         sections[current_section_heading] = {"full_text": full_section_text, "summary": None}
#                     else:
#                         subsection_key = f"{current_section_heading} - {heading_text}"
#                         sections[subsection_key] = {"full_text": full_section_text, "summary": None}

#             # --- REMOVED: Page Count Extraction from GROBID XML ---
#             # page_count = 0
#             # try:
#             #     page_break_tags = soup.find_all('pb')
#             #     ... (rest of old logic removed) ...
#             # except Exception as page_err:
#             #     ...
#             # --- END REMOVED ---

#             print("ExtractionAgent: Finished parsing GROBID XML text.")
#             return {
#                 "title": title,
#                 "sections": sections,
#                 # "page_count": page_count # REMOVED: No longer returned here
#             }
#         except Exception as e:
#             print(f"Error: Failed parsing GROBID XML: {e}")
#             traceback.print_exc()
#             # Return structure with defaults even on error, helps app.py
#             return {"title": "Parsing Error Title", "sections": {}} # REMOVED page_count

#     def _extract_figures_and_tables(self, pdf_path: str) -> Tuple[List[Tuple[str, Optional[bytes]]], List[Tuple[str, Optional[bytes]]]]:
#         """
#         Extracts figures/tables using pdffigures2.
#         Saves output (JSON, images) to a subdirectory within self.pdffigures_output_dir,
#         named after the PDF, containing an 'image' folder and the JSON file.
#         Returns (figures, tables) where each element is (description, image_bytes).
#         """
#         print("ExtractionAgent: Starting figure/table extraction...")
#         figures_list = []
#         tables_list = []

#         if not os.path.exists(self.pdffigures_jar):
#              print(f"Error: pdffigures2 JAR not found at {self.pdffigures_jar}.")
#              return [], []

#         pdf_basename = ntpath.basename(pdf_path)
#         pdf_basename_no_ext = os.path.splitext(pdf_basename)[0]

#         specific_output_dir = os.path.join(self.pdffigures_output_dir, pdf_basename_no_ext) + os.sep
#         os.makedirs(specific_output_dir, exist_ok=True)

#         image_output_dir = os.path.join(specific_output_dir, "image") + os.sep
#         os.makedirs(image_output_dir, exist_ok=True)

#         print(f"ExtractionAgent: PDF-specific output directory: '{specific_output_dir}'")
#         print(f"ExtractionAgent: Image output directory: '{image_output_dir}'")

#         json_output_arg = specific_output_dir
#         image_output_arg = image_output_dir

#         try:
#             cmd = ["java", "-jar", self.pdffigures_jar,
#                    "-d", json_output_arg,
#                    "-m", image_output_arg,
#                    pdf_path]
#             print(f"ExtractionAgent: Running command: {' '.join(cmd)}")

#             cmd_timeout = 300
#             result = subprocess.run(
#                 cmd,
#                 capture_output=True,
#                 text=True,
#                 encoding="utf-8",
#                 errors="replace",
#                 check=False,
#                 timeout=cmd_timeout
#             )

#             if result.stdout:
#                 print("\n--- pdffigures2 STDOUT ---")
#                 print(result.stdout)
#                 print("--- End STDOUT ---")
#             if result.stderr:
#                 print("\n--- pdffigures2 STDERR ---")
#                 print(result.stderr)
#                 print("--- End STDERR ---")

#             if result.returncode != 0:
#                 print(f"Warning: pdffigures2 exited with code {result.returncode}. Figure/Table extraction might be incomplete.")

#             expected_json_filename = f"{pdf_basename_no_ext}.json"
#             json_file_path = os.path.join(specific_output_dir, expected_json_filename)

#             if not os.path.exists(json_file_path):
#                  print(f"Error: pdffigures2 JSON output not found at '{json_file_path}'.")
#                  try: files_in_dir = os.listdir(specific_output_dir); print(f"Files found in '{specific_output_dir}': {files_in_dir}") # Debug
#                  except Exception as list_e: print(f"Could not list files in directory: {list_e}")
#                  return [], []

#             if os.path.getsize(json_file_path) == 0:
#                 print(f"Warning: pdffigures2 JSON file is empty: {json_file_path}")
#                 return [], []

#             try:
#                 with open(json_file_path, "r", encoding="utf-8", errors="replace") as f:
#                     data = json.load(f)
#             except json.JSONDecodeError as e:
#                  print(f"Error: Decoding pdffigures2 JSON failed: {e} in file {json_file_path}")
#                  return [], []

#             if isinstance(data, list):
#                 for item in data:
#                     if not isinstance(item, dict): continue
#                     fig_type = item.get("figType", "Unknown").lower()
#                     caption = item.get("caption", item.get("name", f"Unnamed {fig_type.capitalize()}")).strip()
#                     render_url = item.get("renderURL")

#                     image_binary = None
#                     if render_url:
#                         image_filename = ntpath.basename(render_url)
#                         if image_filename:
#                             image_path = os.path.join(image_output_dir, image_filename)
#                             if os.path.exists(image_path):
#                                 try:
#                                     with open(image_path, "rb") as img_file: image_binary = img_file.read()
#                                 except Exception as e: print(f"Error reading image {image_path}: {e}")
#                             else: print(f"Warning: Image file not found at expected path '{image_path}' for caption '{caption[:50]}...'")
#                         else: print(f"Warning: Could not extract filename from renderURL '{render_url}'")

#                     item_tuple = (caption, image_binary)
#                     if fig_type == "figure": figures_list.append(item_tuple)
#                     elif fig_type == "table": tables_list.append(item_tuple)
#                     else:
#                         print(f"Warning: Unknown figType '{fig_type}'. Classifying as figure.")
#                         figures_list.append(item_tuple)
#             else:
#                  print(f"Warning: Expected list from pdffigures2 JSON, but got {type(data)}. Cannot process.")

#             print(f"ExtractionAgent: Figure/Table data loaded. Figures: {len(figures_list)}, Tables: {len(tables_list)}")
#             print(f"--> Raw pdffigures2 output saved in: '{specific_output_dir}'")
#             return figures_list, tables_list

#         except FileNotFoundError as e:
#             print(f"Error: Java command not found or pdffigures2 JAR path incorrect: {e}")
#             return [], []
#         except subprocess.TimeoutExpired:
#             print(f"Error: pdffigures2 command timed out after {cmd_timeout} seconds.")
#             print(f"Check partial output (if any) in '{specific_output_dir}'")
#             return [], []
#         except Exception as e:
#             print(f"Error: Unexpected error during figure/table extraction: {e}")
#             traceback.print_exc()
#             print(f"Check partial output (if any) in '{specific_output_dir}'")
#             return [], []

#     def _encode_image_data(self, binary_data: Optional[bytes]) -> Optional[str]:
#         """Encodes binary image data to a Base64 string for JSON compatibility."""
#         if binary_data is None: return None
#         try:
#             return base64.b64encode(binary_data).decode('utf-8')
#         except Exception as e:
#             print(f"Error encoding image data to Base64: {e}")
#             return None

#     def _prepare_list_for_json(self, data_list: List[Tuple[str, Optional[bytes]]]) -> List[List[Optional[str]]]:
#         """Converts list of (description, image_bytes) to list of [description, base64_string]."""
#         return [[desc, self._encode_image_data(img_bytes)] for desc, img_bytes in data_list]


#     def extract(self, pdf_path: str) -> Optional[Dict[str, Any]]:
#         """
#         Orchestrates the extraction process for a given PDF.
#         Uses pypdf for page count, GROBID for text structure, and pdffigures2
#         for figures/tables.
#         Returns a dictionary containing title, sections (text), page_count,
#         figures (base64), and tables (base64), or None on failure.
#         """
#         print(f"\n{'='*20} Starting Full Extraction for: {pdf_path} {'='*20}")
#         start_time = time.time()

#         # Ensure the base output directory exists
#         os.makedirs(self.pdffigures_output_dir, exist_ok=True)

#         # 1. Get Page Count directly from PDF using pypdf
#         # This is done first as it's independent and more reliable than GROBID's count
#         page_count = self._get_page_count_from_pdf(pdf_path)
#         if page_count == 0 and os.path.exists(pdf_path):
#             # Log a warning if page count is 0 but the file exists (might be unreadable PDF)
#             print("Extraction Agent: Warning - pypdf reported 0 pages. The PDF might be unreadable or corrupted.")
#         elif not os.path.exists(pdf_path):
#              print("Extraction Agent: Error - PDF file not found. Cannot proceed.")
#              return None # Cannot proceed if PDF doesn't exist

#         # 2. Get Text Structure from Grobid
#         xml_content = self._get_grobid_xml(pdf_path)
#         if not xml_content:
#             print("Extraction Agent: GROBID processing failed or returned no content. Aborting text extraction part.")
#             # Decide if you want to proceed without text or abort entirely
#             # Let's proceed but use default values for text parts
#             text_data = {"title": "GROBID Failed", "sections": {}}
#         else:
#             text_data = self._parse_grobid_xml(xml_content)
#             # Check if parsing itself failed catastrophically
#             if not text_data or text_data.get("title") == "Parsing Error Title":
#                 print("Extraction Agent: Failed to parse GROBID XML structure. Using default text values.")
#                 # Ensure text_data is a dictionary even if parsing returns None
#                 text_data = {"title": "GROBID Parsing Failed", "sections": {}}

#         # Check for a meaningful title (allow proceeding without one but log warning)
#         if text_data.get("title", "").startswith("No title found") or \
#            text_data.get("title", "").startswith("GROBID") or \
#            text_data.get("title", "").startswith("Parsing"):
#              print(f"Extraction Agent: Warning - Proceeding with potentially invalid title: '{text_data.get('title')}'")

#         print(f"Extraction Agent: Text structure extracted (or defaults used). Title: '{text_data.get('title', 'N/A')[:60]}...'")


#         # 3. Extract Figures and Tables using pdffigures2 (saves to persistent dir)
#         figures_binary, tables_binary = self._extract_figures_and_tables(pdf_path)

#         # 4. Prepare figures/tables for JSON output (Base64 encoding)
#         print("ExtractionAgent: Encoding images to Base64...")
#         figures_for_json = self._prepare_list_for_json(figures_binary)
#         tables_for_json = self._prepare_list_for_json(tables_binary)

#         # 5. Combine all results
#         result = {
#             "title": text_data.get("title", "Extraction Error Title"), # Use default if missing
#             "sections": text_data.get("sections", {}),
#             "page_count": page_count, # <--- Use page count from pypdf
#             "figures": figures_for_json,
#             "tables": tables_for_json
#         }

#         end_time = time.time()
#         pdf_basename_no_ext = os.path.splitext(ntpath.basename(pdf_path))[0]
#         final_output_dir = os.path.join(self.pdffigures_output_dir, pdf_basename_no_ext)
#         print(f"\nExtraction Agent: Extraction process completed in {end_time - start_time:.2f} seconds.")
#         print(f"  - Page Count (from pypdf): {page_count}")
#         print(f"  - Figures Found: {len(figures_for_json)}")
#         print(f"  - Tables Found: {len(tables_for_json)}")
#         print(f"Check raw pdffigures2 output in: '{final_output_dir}{os.sep}'")
#         print(f"{'='*70}\n")
#         return result




import os
import requests
from bs4 import BeautifulSoup
import subprocess
import json
import time
import ntpath
import traceback
import base64
import re
from typing import List, Tuple, Optional, Dict, Any, TypedDict

import pypdf
from langgraph.graph import StateGraph, END

# --- Helper functions ---
def _encode_image_data(binary_data: Optional[bytes]) -> Optional[str]:
    if binary_data is None: return None
    try:
        return base64.b64encode(binary_data).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image data to Base64: {e}")
        return None

def _prepare_list_for_json(data_list: List[Tuple[str, Optional[bytes]]]) -> List[List[Optional[str]]]:
    return [[desc, _encode_image_data(img_bytes)] for desc, img_bytes in data_list]

# --- State Definition for LangGraph ---
class ExtractionState(TypedDict):
    pdf_path: str
    grobid_url: str
    pdffigures_jar: str
    pdffigures_output_dir_base: str

    page_count: Optional[int]
    grobid_xml_content: Optional[str]
    text_data: Optional[Dict[str, Any]]
    figures_binary: Optional[List[Tuple[str, Optional[bytes]]]]
    tables_binary: Optional[List[Tuple[str, Optional[bytes]]]]
    final_output: Optional[Dict[str, Any]]
    error_message: Optional[str]

class ExtractionAgentLangGraph:
    def __init__(self, grobid_url: str, pdffigures_jar: str, pdffigures_output_dir: str = "pdffigures2_output_lg"):
        self.grobid_url = grobid_url
        self.pdffigures_jar = pdffigures_jar
        self.pdffigures_output_dir_base = pdffigures_output_dir # This is the base for all PDFs

        if not os.path.exists(self.pdffigures_jar):
            print(f"WARNING: pdffigures2 JAR not found at {self.pdffigures_jar}. Figure/Table extraction will fail.")
        else:
            print(f"ExtractionAgentLangGraph: Found pdffigures2 JAR at {self.pdffigures_jar}")
        print(f"ExtractionAgentLangGraph: Using GROBID API at {self.grobid_url}")
        print(f"ExtractionAgentLangGraph: pdffigures2 output base directory: '{self.pdffigures_output_dir_base}'")

        os.makedirs(self.pdffigures_output_dir_base, exist_ok=True)

        self.workflow = StateGraph(ExtractionState)
        self._build_graph()
        self.app = self.workflow.compile()

    def _get_page_count_from_pdf_node(self, state: ExtractionState) -> Dict[str, Any]:
        pdf_path = state["pdf_path"]
        print(f"Node: Reading page count from '{pdf_path}' using pypdf...")
        page_count = 0
        error_message = state.get("error_message") # Preserve existing errors
        
        if not os.path.exists(pdf_path):
            err_msg = f"PDF file not found at '{pdf_path}' for page count."
            print(f"Error: {err_msg}")
            error_message = f"{error_message}; {err_msg}" if error_message else err_msg
            # This is a critical error; the graph should stop or handle it gracefully.
            # The _should_continue conditional edge will handle routing.
            return {"page_count": 0, "error_message": error_message}

        try:
            reader = pypdf.PdfReader(pdf_path)
            page_count = len(reader.pages)
            print(f"Node: pypdf reported {page_count} pages.")
        except pypdf.errors.PdfReadError as pdf_err:
            err_msg = f"pypdf could not read PDF '{pdf_path}'. Corrupted/encrypted? Error: {pdf_err}"
            print(f"Error: {err_msg}")
            error_message = f"{error_message}; {err_msg}" if error_message else err_msg
        except Exception as e:
            err_msg = f"Unexpected error getting page count with pypdf: {e}"
            print(f"Error: {err_msg}")
            traceback.print_exc()
            error_message = f"{error_message}; {err_msg}" if error_message else err_msg

        if page_count == 0 and os.path.exists(pdf_path) and "pypdf could not read" not in (error_message or ""):
             print("Node: Warning - pypdf reported 0 pages. PDF might be unreadable or empty.")

        return {"page_count": page_count, "error_message": error_message}

    def _get_grobid_xml_node(self, state: ExtractionState) -> Dict[str, Any]:
        pdf_path = state["pdf_path"]
        grobid_url = state["grobid_url"]
        print("Node: Requesting GROBID processing...")
        xml_content = None
        error_message = state.get("error_message")

        # No need to check for "PDF file not found" here, _should_continue handles it.
        
        try:
            with open(pdf_path, "rb") as f:
                response = requests.post(
                    grobid_url, files={"input": f},
                    data={"consolidateHeader": "1", "consolidateCitations": "0"}, timeout=300
                )
                response.raise_for_status()
                print("Node: GROBID request successful.")
                xml_content = response.text
        except requests.exceptions.Timeout:
            err_msg = "GROBID request timed out."
            print(f"Error: {err_msg}")
            error_message = f"{error_message}; {err_msg}" if error_message else err_msg
        except requests.exceptions.RequestException as e:
            err_msg = f"GROBID connection/request failed (Is server at {grobid_url} running?): {e}"
            print(f"Error: {err_msg}")
            error_message = f"{error_message}; {err_msg}" if error_message else err_msg
        except Exception as e: # Catches FileNotFoundError if somehow missed by page count node
            err_msg = f"Unexpected error during GROBID request (PDF path: {pdf_path}): {e}"
            print(f"Error: {err_msg}")
            traceback.print_exc()
            error_message = f"{error_message}; {err_msg}" if error_message else err_msg
        return {"grobid_xml_content": xml_content, "error_message": error_message}

    def _parse_grobid_xml_node(self, state: ExtractionState) -> Dict[str, Any]:
        xml_content = state["grobid_xml_content"]
        print("Node: Parsing GROBID XML for text structure...")
        error_message = state.get("error_message")
        # Default in case of parsing failure or no XML content
        text_data = {"title": "GROBID Processing Incomplete", "sections": {}} 

        if not xml_content:
            print("Node: Cannot parse empty GROBID XML content. Using defaults for text_data.")
            if "GROBID XML content was empty" not in (error_message or "") and \
               "GROBID request timed out" not in (error_message or "") and \
               "GROBID connection/request failed" not in (error_message or ""): # Avoid redundant errors
                err_msg = "GROBID XML content was empty (or previous GROBID step failed)."
                error_message = f"{error_message}; {err_msg}" if error_message else err_msg
            return {"text_data": text_data, "error_message": error_message}

        try:
            soup = BeautifulSoup(xml_content, "lxml-xml")
            title_tag = soup.select_one("teiHeader > fileDesc > titleStmt > title")
            title = title_tag.get_text(strip=True) if title_tag else "No title found"

            abstract_tag = soup.select_one("teiHeader > profileDesc > abstract")
            abstract = abstract_tag.get_text(strip=True) if abstract_tag else ""

            sections = {} # Standard dict is fine for LangGraph internal processing
            if abstract:
                sections["Abstract"] = {"full_text": abstract, "summary": None}

            body = soup.select_one("text > body")
            if body:
                # ... (rest of your parsing logic for sections)
                top_level_keywords = ["abstract", "background", "introduction", "related work", "method", "methodology", "materials and methods", "experiment", "results", "discussion", "conclusion", "acknowledgement", "reference", "appendix", "supplementary"]
                current_section_heading = "Introduction"; section_counter = 1
                for div in body.find_all("div", recursive=False):
                    head = div.find("head"); heading_text = "Unnamed Section"
                    if head:
                        heading_text = re.sub(r"^\s*(\d+\.|\w\.|[IVXLCDM]+\.)\s*", "", head.get_text(strip=True)).strip()
                        if not heading_text: heading_text = f"Section {head.get('n', section_counter)}"; section_counter +=1
                    else: heading_text = f"Unnamed Section {section_counter}"; section_counter += 1
                    is_major_section = any(keyword in heading_text.lower() for keyword in top_level_keywords)
                    paragraphs = div.find_all("p", recursive=False)
                    div_text_nodes = div.find_all(string=True, recursive=False)
                    direct_text = "\n".join(t.strip() for t in div_text_nodes if t.strip() and not t.parent.name in ['head', 'figure', 'table'])
                    para_text = "\n".join(p.get_text(strip=True) for p in paragraphs)
                    full_section_text = (para_text + "\n" + direct_text).strip()
                    if not full_section_text: continue
                    if is_major_section: current_section_heading = heading_text; sections[current_section_heading] = {"full_text": full_section_text, "summary": None}
                    else: sections[f"{current_section_heading} - {heading_text}"] = {"full_text": full_section_text, "summary": None}
            
            text_data = {"title": title, "sections": sections}
            print("Node: Finished parsing GROBID XML text.")
            if text_data.get("title", "").startswith("No title found"):
                print(f"Node: Warning - Proceeding with potentially invalid title: '{title}'")
        except Exception as e:
            err_msg = f"Failed parsing GROBID XML: {e}"
            print(f"Error: {err_msg}"); traceback.print_exc()
            error_message = f"{error_message}; {err_msg}" if error_message else err_msg
            text_data = {"title": "Parsing Error Title", "sections": {}} # Fallback
        
        return {"text_data": text_data, "error_message": error_message}

    def _extract_figures_and_tables_node(self, state: ExtractionState) -> Dict[str, Any]:
        pdf_path = state["pdf_path"]
        pdffigures_jar = state["pdffigures_jar"]
        pdffigures_output_dir_base = state["pdffigures_output_dir_base"]
        print("Node: Starting figure/table extraction...")
        figures_list_binary, tables_list_binary = [], []
        error_message = state.get("error_message")
        
        # No need to check for "PDF file not found" here, _should_continue handles it.

        if not os.path.exists(pdffigures_jar):
            err_msg = f"pdffigures2 JAR not found at {pdffigures_jar}."
            print(f"Error: {err_msg}")
            error_message = f"{error_message}; {err_msg}" if error_message else err_msg
            return {"figures_binary": [], "tables_binary": [], "error_message": error_message}

        pdf_basename_no_ext = os.path.splitext(ntpath.basename(pdf_path))[0]
        specific_output_dir = os.path.join(pdffigures_output_dir_base, pdf_basename_no_ext) + os.sep
        os.makedirs(specific_output_dir, exist_ok=True)
        image_output_dir = os.path.join(specific_output_dir, "image") + os.sep
        os.makedirs(image_output_dir, exist_ok=True)
        print(f"Node: PDF-specific output dir for pdffigures2: '{specific_output_dir}'")

        try:
            cmd = ["java", "-jar", pdffigures_jar, "-d", specific_output_dir, "-m", image_output_dir, pdf_path]
            print(f"Node: Running command: {' '.join(cmd)}")
            cmd_timeout = 300
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=cmd_timeout)
            if result.stdout: print(f"\n--- pdffigures2 STDOUT ---\n{result.stdout}\n--- End STDOUT ---")
            if result.stderr: print(f"\n--- pdffigures2 STDERR ---\n{result.stderr}\n--- End STDERR ---")
            if result.returncode != 0: print(f"Warning: pdffigures2 exited with code {result.returncode}.")

            json_file_path = os.path.join(specific_output_dir, f"{pdf_basename_no_ext}.json")
            if not os.path.exists(json_file_path):
                err_msg = f"pdffigures2 JSON output not found at '{json_file_path}'. Output dir content: {os.listdir(specific_output_dir) if os.path.exists(specific_output_dir) else 'Not Found'}"
                print(f"Error: {err_msg}")
                error_message = f"{error_message}; {err_msg}" if error_message else err_msg
                return {"figures_binary": [], "tables_binary": [], "error_message": error_message}
            if os.path.getsize(json_file_path) == 0:
                print(f"Warning: pdffigures2 JSON file is empty: {json_file_path}")
                return {"figures_binary": [], "tables_binary": [], "error_message": error_message} # No new error, just empty
            
            with open(json_file_path, "r", encoding="utf-8", errors="replace") as f: data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict): continue
                    fig_type = item.get("figType", "Unknown").lower()
                    caption = item.get("caption", item.get("name", f"Unnamed {fig_type.capitalize()}")).strip()
                    render_url = item.get("renderURL"); image_binary = None
                    if render_url:
                        image_filename = ntpath.basename(render_url)
                        if image_filename:
                            image_path = os.path.join(image_output_dir, image_filename)
                            if os.path.exists(image_path):
                                try:
                                    with open(image_path, "rb") as img_file: image_binary = img_file.read()
                                except Exception as e_read: print(f"Error reading image {image_path}: {e_read}")
                            else: print(f"Warning: Image file not found: '{image_path}'")
                    item_tuple = (caption, image_binary)
                    if fig_type == "figure": figures_list_binary.append(item_tuple)
                    elif fig_type == "table": tables_list_binary.append(item_tuple)
                    else: figures_list_binary.append(item_tuple) # Default to figure
            else: print(f"Warning: Expected list from pdffigures2 JSON, got {type(data)}.")
            print(f"Node: Figure/Table data loaded. Figures: {len(figures_list_binary)}, Tables: {len(tables_list_binary)}")
            print(f"--> Raw pdffigures2 output saved in: '{specific_output_dir}'")
        except FileNotFoundError as e: # Java not found
            err_msg = f"Java cmd not found or pdffigures2 JAR path incorrect: {e}"
            print(f"Error: {err_msg}"); error_message = f"{error_message}; {err_msg}" if error_message else err_msg
        except subprocess.TimeoutExpired:
            err_msg = f"pdffigures2 timed out after {cmd_timeout}s."
            print(f"Error: {err_msg}"); error_message = f"{error_message}; {err_msg}" if error_message else err_msg
        except json.JSONDecodeError as e_json:
            err_msg = f"Decoding pdffigures2 JSON failed: {e_json} in file {json_file_path}"
            print(f"Error: {err_msg}"); error_message = f"{error_message}; {err_msg}" if error_message else err_msg
        except Exception as e:
            err_msg = f"Unexpected error during figure/table extraction: {e}"
            print(f"Error: {err_msg}"); traceback.print_exc(); error_message = f"{error_message}; {err_msg}" if error_message else err_msg
        
        return {"figures_binary": figures_list_binary, "tables_binary": tables_list_binary, "error_message": error_message}

    def _combine_results_node(self, state: ExtractionState) -> Dict[str, Any]:
        print("Node: Combining all results and encoding images...")
        # Provide defaults if keys are missing from state (e.g., due to early exit or error)
        text_data = state.get("text_data")
        # Ensure text_data is a dict with default title/sections if it's None
        if text_data is None:
            title_from_error = "Extraction Error Title"
            if state.get("error_message"):
                if "PDF file not found" in state["error_message"]: title_from_error = "PDF Not Found"
                elif "pypdf could not read" in state["error_message"]: title_from_error = "Unreadable PDF"
            text_data = {"title": title_from_error, "sections": {}}

        page_count = state.get("page_count", 0) # Default to 0 if not set
        figures_binary = state.get("figures_binary", [])
        tables_binary = state.get("tables_binary", [])

        figures_for_json = _prepare_list_for_json(figures_binary)
        tables_for_json = _prepare_list_for_json(tables_binary)
        
        final_result = {
            "title": text_data.get("title", "Extraction Error Title"), # Final fallback
            "sections": text_data.get("sections", {}),
            "page_count": page_count,
            "figures": figures_for_json,
            "tables": tables_for_json,
            "extraction_errors": state.get("error_message") # Pass along any accumulated errors
        }
        print("Node: Results combined.")
        # No need to return error_message separately if it's in final_output
        return {"final_output": final_result}


    def _should_continue(self, state: ExtractionState) -> str:
        """Conditional edge: if PDF not found or unreadable by pypdf, go to combine."""
        error_msg = state.get("error_message")
        if error_msg:
            if "PDF file not found" in error_msg or "pypdf could not read PDF" in error_msg:
                print(f"Conditional Edge: Critical PDF error ('{error_msg}'), routing to combine_results.")
                return "combine_results" 
        return "get_grobid_xml" # Default path

    def _build_graph(self):
        self.workflow.add_node("get_page_count", self._get_page_count_from_pdf_node)
        self.workflow.add_node("get_grobid_xml", self._get_grobid_xml_node)
        self.workflow.add_node("parse_grobid_xml", self._parse_grobid_xml_node)
        self.workflow.add_node("extract_figures_tables", self._extract_figures_and_tables_node)
        self.workflow.add_node("combine_results", self._combine_results_node)

        self.workflow.set_entry_point("get_page_count")
        self.workflow.add_conditional_edges(
            "get_page_count",
            self._should_continue,
            {
                "get_grobid_xml": "get_grobid_xml",
                "combine_results": "combine_results" 
            }
        )
        self.workflow.add_edge("get_grobid_xml", "parse_grobid_xml")
        self.workflow.add_edge("parse_grobid_xml", "extract_figures_tables")
        self.workflow.add_edge("extract_figures_tables", "combine_results")
        self.workflow.add_edge("combine_results", END)

    def extract(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        print(f"\n{'='*20} LANGGRAPH: Starting Full Extraction for: {pdf_path} {'='*20}")
        start_time = time.time()

        initial_state: ExtractionState = {
            "pdf_path": pdf_path,
            "grobid_url": self.grobid_url,
            "pdffigures_jar": self.pdffigures_jar,
            "pdffigures_output_dir_base": self.pdffigures_output_dir_base,
            "page_count": None, "grobid_xml_content": None, "text_data": None,
            "figures_binary": None, "tables_binary": None, "final_output": None,
            "error_message": None
        }
        
        final_state = self.app.invoke(initial_state)
        result = final_state.get("final_output")
        end_time = time.time()
        
        if not result: # Should always have final_output from combine_results
            result = {
                "title": "Critical Graph Invocation Error", "sections": {}, "page_count": 0,
                "figures": [], "tables": [],
                "extraction_errors": final_state.get("error_message", "Unknown critical graph failure, final_output missing.")
            }

        pdf_basename_no_ext = os.path.splitext(ntpath.basename(pdf_path))[0]
        final_pdffigures_dir_for_pdf = os.path.join(self.pdffigures_output_dir_base, pdf_basename_no_ext)

        print(f"\nExtractionAgentLangGraph: Extraction process completed in {end_time - start_time:.2f} seconds.")
        print(f"  - Page Count: {result.get('page_count', 'N/A')}")
        print(f"  - Title: '{str(result.get('title', 'N/A'))[:60]}...'")
        print(f"  - Figures: {len(result.get('figures', []))}, Tables: {len(result.get('tables', []))}")
        if result.get("extraction_errors"):
            print(f"  - EXTRACTION ERRORS: {result['extraction_errors']}")
        
        # Only point to pdffigures dir if it was likely attempted
        if not (result.get("extraction_errors") and ("PDF file not found" in result["extraction_errors"] or "pypdf could not read" in result["extraction_errors"])):
             print(f"Check raw pdffigures2 output (if generated) in: '{final_pdffigures_dir_for_pdf}{os.sep}'")
        else:
             print(f"pdffigures2 was skipped due to critical PDF error.")
        print(f"{'='*70}\n")
        return result