# import time
# import traceback
# import re
# from typing import Optional, List
# import math # For ceiling function

# # Optional: NLTK for sentence splitting (if desired, uncomment and install)
# # try:
# #     import nltk
# #     nltk.download('punkt', quiet=True)
# #     NLTK_AVAILABLE = True
# # except ImportError:
# #     print("INFO: 'nltk' not found. Using basic regex for sentence splitting.")
# #     NLTK_AVAILABLE = False

# from utils import clean_text_for_summarization # Import shared cleaning function
# from config import TOGETHER_API_KEY, DEFAULT_SUMMARY_REFINEMENT_MODEL_NAME # Import Together AI config

# # --- Together.ai Integration ---
# try:
#     from together import Together
#     TOGETHER_AVAILABLE = True
# except ImportError:
#     print("WARNING: 'together' library not found. LLM summary features will be unavailable.")
#     print("Install with: pip install together")
#     Together = None
#     TOGETHER_AVAILABLE = False
# # --- End Together.ai Integration ---


# class SummarizationAgent:
#     """
#     Agent responsible for summarizing text using a pre-loaded summarization pipeline.
#     Includes text cleaning, generation parameter tuning, and automatic chunking
#     for long inputs. For chunked inputs, it summarizes chunks locally, then uses
#     Together AI (if available) to synthesize the final summary from chunk summaries,
#     otherwise falls back to local summarization for the final step.
#     For short inputs, it summarizes locally and optionally refines with Together AI.
#     """
#     def __init__(self, pipeline_instance: Optional[object], device_name: str):
#         """
#         Initializes the agent.
#         :param pipeline_instance: A loaded Hugging Face summarization pipeline object.
#         :param device_name: Name of the device the pipeline is running on (e.g., "CPU", "GPU 0").
#         """
#         self.pipeline = pipeline_instance
#         self.device_name = device_name
#         self.is_local_pipeline_available = pipeline_instance is not None

#         # --- Tokenizer and Max Length ---
#         self.tokenizer = None
#         self.model_max_input_tokens = 1024 # Default
#         if self.is_local_pipeline_available and hasattr(self.pipeline, 'tokenizer'):
#             self.tokenizer = self.pipeline.tokenizer
#             if hasattr(self.tokenizer, 'model_max_length'):
#                 self.model_max_input_tokens = self.tokenizer.model_max_length
#                 print(f"Summarization Agent: Detected model max input tokens: {self.model_max_input_tokens}")
#             else:
#                  print(f"Summarization Agent WARNING: Cannot detect model_max_length. Using default: {self.model_max_input_tokens}")
#         elif self.is_local_pipeline_available:
#             print(f"Summarization Agent WARNING: Pipeline object found, but no tokenizer. Using default max tokens: {self.model_max_input_tokens}")

#         # --- Chunking Parameters ---
#         self.chunk_token_buffer = 50
#         self.max_chunk_tokens = max(100, self.model_max_input_tokens - self.chunk_token_buffer)
#         self.chunk_overlap_ratio = 0.15
#         self.min_chunk_tokens = 50

#         # --- Summarization Length Parameters ---
#         self.final_summary_max_len = 350 # Target length for the *final* output
#         self.final_summary_min_len = 50
#         self.chunk_summary_max_len = max(50, int(self.final_summary_max_len * 0.6))
#         self.chunk_summary_min_len = max(20, int(self.final_summary_min_len * 0.6))
#         # Max tokens for LLM synthesis prompt (can be larger than final output length)
#         self.llm_synthesis_max_tokens = self.final_summary_max_len + 150

#         # --- Initialize Together.ai client ---
#         self.together_client = None
#         self.is_llm_available = False # Renamed for clarity (covers refinement & synthesis)
#         self.refinement_model_name = DEFAULT_SUMMARY_REFINEMENT_MODEL_NAME

#         if TOGETHER_AVAILABLE and Together:
#             if not TOGETHER_API_KEY or TOGETHER_API_KEY == "YOUR_TOGETHER_API_KEY_PLACEHOLDER":
#                  print("WARNING: TOGETHER_API_KEY not set. Together AI features unavailable.")
#             else:
#                 try:
#                     self.together_client = Together(api_key=TOGETHER_API_KEY)
#                     self.is_llm_available = True
#                     print(f"Together AI client initialized (Model for refinement/synthesis: {self.refinement_model_name}).")
#                 except Exception as e:
#                     print(f"ERROR initializing Together AI client: {e}")
#                     traceback.print_exc()
#                     self.together_client = None
#         else:
#              print("Together AI library not available. LLM features disabled.")
#         # --- End Together.ai Initialization ---

#         print(f"\n--- Summarization Agent Status ---")
#         print(f"Local Summarization Pipeline Available: {self.is_local_pipeline_available} (on {self.device_name})")
#         print(f"--> Model Max Input Tokens: {self.model_max_input_tokens}")
#         print(f"--> Effective Max Chunk Tokens: {self.max_chunk_tokens}")
#         print(f"Together AI LLM Available (for refinement/synthesis): {self.is_llm_available}")
#         print("-" * 34)

#     # --- Helper Methods (Tokenization, Chunking) ---
#     def _tokenize(self, text: str, add_special_tokens=False) -> List[int]:
#         if not self.tokenizer: return list(range(len(text.split()))) # Fallback
#         try:
#             return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
#         except Exception: return list(range(len(text.split())))

#     def _get_token_count(self, text: str) -> int:
#         return len(self._tokenize(text, add_special_tokens=False))

#     def _chunk_text_by_tokens(self, text: str) -> List[str]:
#         # (Chunking logic remains the same as in the previous version)
#         # ... [Keep the exact code from the previous _chunk_text_by_tokens method here] ...
#         if not text: return []
#         print(f"Summarization Agent: Chunking text (length {len(text)} chars)...")
#         overlap_tokens = int(self.max_chunk_tokens * self.chunk_overlap_ratio)
#         sentences = re.split(r'(?<=[.!?])\s+', text.strip())
#         sentences = [s for s in sentences if s]
#         if not sentences:
#             print("Summarization Agent WARNING: Could not split into sentences. Falling back to paragraphs.")
#             sentences = text.split('\n\n')
#             sentences = [s.strip() for s in sentences if s.strip()]
#             if not sentences:
#                 print("Summarization Agent WARNING: Could not split meaningfully. Treating as single chunk.")
#                 single_chunk_tokens = self._get_token_count(text)
#                 if single_chunk_tokens > self.max_chunk_tokens:
#                     print(f"Summarization Agent ERROR: Single un-splittable chunk ({single_chunk_tokens} tokens) exceeds max chunk size ({self.max_chunk_tokens}). Cannot proceed.")
#                     return [] # Signal failure
#                 else:
#                     return [text]

#         chunks = []
#         current_chunk_sentences = []
#         current_token_count = 0
#         overlap_sentence_buffer = []

#         for i, sentence in enumerate(sentences):
#             sentence = sentence.strip()
#             if not sentence: continue
#             sentence_token_count = self._get_token_count(sentence)

#             if sentence_token_count > self.max_chunk_tokens:
#                 print(f"Summarization Agent WARNING: Sentence {i+1} ('{sentence[:80]}...') too long ({sentence_token_count} > {self.max_chunk_tokens}). Adding as separate chunk (may be truncated by pipeline).")
#                 if current_chunk_sentences:
#                     chunk_text = " ".join(current_chunk_sentences)
#                     if self._get_token_count(chunk_text) >= self.min_chunk_tokens: chunks.append(chunk_text)
#                 chunks.append(sentence)
#                 current_chunk_sentences = []
#                 current_token_count = 0
#                 overlap_sentence_buffer = []
#                 continue

#             if current_token_count + sentence_token_count <= self.max_chunk_tokens:
#                 current_chunk_sentences.append(sentence)
#                 current_token_count += sentence_token_count
#                 overlap_sentence_buffer.append(sentence)
#             else:
#                 if current_chunk_sentences:
#                     chunk_text = " ".join(current_chunk_sentences)
#                     if self._get_token_count(chunk_text) >= self.min_chunk_tokens: chunks.append(chunk_text)
#                     else: print(f"Summarization Agent INFO: Discarding short potential chunk ending sentence {i}.")

#                 overlap_sentences_to_keep = []
#                 overlap_token_count_actual = 0
#                 for overlap_sent in reversed(overlap_sentence_buffer):
#                      overlap_sent_tokens = self._get_token_count(overlap_sent)
#                      if overlap_token_count_actual + overlap_sent_tokens <= overlap_tokens:
#                           overlap_sentences_to_keep.insert(0, overlap_sent)
#                           overlap_token_count_actual += overlap_sent_tokens
#                      else: break

#                 new_chunk_initial_sentences = overlap_sentences_to_keep + [sentence]
#                 current_chunk_sentences = new_chunk_initial_sentences
#                 current_token_count = self._get_token_count(" ".join(current_chunk_sentences))
#                 overlap_sentence_buffer = list(current_chunk_sentences)

#                 if current_token_count > self.max_chunk_tokens:
#                     print(f"Summarization Agent WARNING: New chunk starting overlap + sentence {i+1} ('{sentence[:50]}...') too long ({current_token_count}). Starting chunk with just the sentence.")
#                     current_chunk_sentences = [sentence]
#                     current_token_count = sentence_token_count
#                     overlap_sentence_buffer = [sentence]

#         if current_chunk_sentences:
#             last_chunk_text = " ".join(current_chunk_sentences)
#             if self._get_token_count(last_chunk_text) >= self.min_chunk_tokens: chunks.append(last_chunk_text)
#             elif not chunks: chunks.append(last_chunk_text)

#         print(f"Summarization Agent: Created {len(chunks)} chunks.")
#         return chunks


#     # --- Core Summarization Methods ---
#     def _summarize_single_chunk(self, chunk_text: str, chunk_index: int, total_chunks: int) -> Optional[str]:
#         """Summarizes a single text chunk using the local pipeline."""
#         if not self.is_local_pipeline_available: return None
#         print(f"Summarization Agent: Summarizing chunk {chunk_index + 1}/{total_chunks} locally ({len(chunk_text)} chars, {self._get_token_count(chunk_text)} tokens)...")
#         try:
#             start_time = time.time()
#             summary_output = self.pipeline(
#                 chunk_text,
#                 max_length=self.chunk_summary_max_len,
#                 min_length=self.chunk_summary_min_len,
#                 do_sample=False, truncation=True, no_repeat_ngram_size=3, early_stopping=True,
#             )
#             end_time = time.time()
#             summary = summary_output[0]['summary_text'].strip() if summary_output and summary_output[0].get('summary_text') else None
#             if not summary: print(f"Summarization Agent WARNING: Empty summary for chunk {chunk_index + 1}.")
#             else: print(f"Summarization Agent: Chunk {chunk_index + 1} summarized locally in {end_time - start_time:.2f}s.")
#             return summary
#         except Exception as e:
#             print(f"Summarization Agent ERROR summarizing chunk {chunk_index + 1} locally: {e}")
#             # Optionally add traceback print here
#             return None

#     def _summarize_combined_locally(self, combined_text: str) -> str:
#         """Summarizes combined text using the local pipeline (fallback/final step)."""
#         if not self.is_local_pipeline_available:
#             return "Summarization failed: Local pipeline unavailable for final combination."

#         print(f"Summarization Agent: Summarizing combined text locally (fallback/final step)...")
#         combined_tokens = self._get_token_count(combined_text)
#         if combined_tokens > self.model_max_input_tokens:
#              print(f"Summarization Agent WARNING: Combined text ({combined_tokens} tokens) exceeds local model max ({self.model_max_input_tokens}). Truncation will occur.")

#         try:
#             start_time = time.time()
#             final_output = self.pipeline(
#                 combined_text,
#                 max_length=self.final_summary_max_len,
#                 min_length=self.final_summary_min_len,
#                 do_sample=False, truncation=True, no_repeat_ngram_size=3, early_stopping=True,
#             )
#             end_time = time.time()
#             summary = final_output[0]['summary_text'].strip() if final_output and final_output[0].get('summary_text') else None
#             if summary:
#                 print(f"Summarization Agent: Local final summary generated in {end_time - start_time:.2f}s.")
#                 return summary
#             else:
#                 print("Summarization Agent WARNING: Local final summarization resulted in empty output.")
#                 return "Summarization failed: Empty output during final local combination."
#         except Exception as e:
#             print(f"Summarization Agent ERROR during final local summarization: {e}")
#             # traceback.print_exc() # Uncomment for debugging
#             err_str = str(e).lower()
#             if "out of memory" in err_str: return "Summarization failed: Out of memory during final local combination."
#             elif "maximum sequence length" in err_str: return f"Summarization failed: Combined text exceeds local model max length ({self.model_max_input_tokens}) during final step."
#             else: return f"Summarization failed (Final Local Step: {type(e).__name__})."

#     # --- Together AI Interaction ---
#     def _call_together_ai(self, prompt: str, max_tokens: int) -> Optional[str]:
#         """Helper function to call the Together AI API."""
#         if not self.is_llm_available or not self.together_client:
#             print("Summarization Agent: Cannot call Together AI, client not available.")
#             return None

#         print(f"Summarization Agent: Sending request to Together AI (Model: {self.refinement_model_name})...")
#         try:
#             start_time = time.time()
#             response = self.together_client.completions.create(
#                 prompt=prompt,
#                 model=self.refinement_model_name,
#                 max_tokens=max_tokens,
#                 temperature=0.5,
#                 top_p=0.8,
#                 repetition_penalty=1.05,
#                 stop=["\n\nHuman:", "</s>", "<|endoftext|>", "---"]
#             )
#             end_time = time.time()

#             if response and hasattr(response, 'choices') and response.choices:
#                  if hasattr(response.choices[0], 'text'):
#                      result_text = response.choices[0].text.strip()
#                      # Clean potential stop sequences
#                      for stop_seq in ["\n\nHuman:", "</s>", "<|endoftext|>", "---"]:
#                           if result_text.endswith(stop_seq):
#                                result_text = result_text[:-len(stop_seq)].strip()

#                      if result_text:
#                         print(f"Summarization Agent: Received response from LLM in {end_time - start_time:.2f} seconds.")
#                         return result_text
#                      else:
#                         print("Summarization Agent WARNING: LLM response was empty.")
#                         return None
#                  else:
#                       print("Summarization Agent WARNING: LLM response choice missing 'text' attribute.")
#                       return None
#             else:
#                  print("Summarization Agent WARNING: LLM response structure unexpected or empty.")
#                  return None

#         except Exception as e:
#             print(f"Summarization Agent ERROR calling Together AI API: {e}")
#             if hasattr(e, 'response'):
#                  try:
#                       print(f"API Response Status: {e.response.status_code}")
#                       print(f"API Response Body: {e.response.text}")
#                  except Exception: pass
#             # traceback.print_exc() # Uncomment for debugging API errors
#             return None

#     def _refine_summary_with_llm(self, original_summary: str) -> Optional[str]:
#         """Uses Together AI LLM to REFINE a given summary (typically from direct local summarization)."""
#         refinement_prompt = (
#             f"You are an expert editor. Refine the following summary for clarity, flow, and conciseness.\n"
#             f"Correct grammar and awkward phrasing. Remove redundancy. Preserve all key information.\n"
#             f"Output ONLY the refined summary text.\n\n"
#             f"--- ORIGINAL SUMMARY ---\n{original_summary}\n--- END ORIGINAL SUMMARY ---\n\n"
#             f"REFINED SUMMARY:"
#         )
#         # Use final summary length constraints, allow some buffer
#         return self._call_together_ai(refinement_prompt, self.final_summary_max_len + 100)

#     def _synthesize_summary_with_llm(self, combined_chunk_summaries: str) -> Optional[str]:
#         """Uses Together AI LLM to SYNTHESIZE a final summary from combined chunk summaries."""
#         synthesis_prompt = (
#             f"You are an expert summarizer. Synthesize the following intermediate summaries (from sequential parts of a document) into a single, coherent final summary.\n"
#             f"Capture the most important information from all parts, maintain logical flow, and eliminate redundancy between the summaries.\n"
#             f"Output ONLY the final synthesized summary text.\n\n"
#             f"--- INTERMEDIATE SUMMARIES ---\n{combined_chunk_summaries}\n--- END INTERMEDIATE SUMMARIES ---\n\n"
#             f"FINAL SUMMARY:"
#         )
#         # Use the dedicated max tokens setting for synthesis
#         return self._call_together_ai(synthesis_prompt, self.llm_synthesis_max_tokens)

#     # --- Main Summarization Method ---
#     def summarize(self, text_to_summarize: str) -> str:
#         """
#         Summarizes text using the appropriate workflow: direct + optional LLM refinement
#         for short texts, or chunking + local chunk summaries + LLM/local synthesis for long texts.
#         """
#         start_overall_time = time.time()

#         # --- Input Validation and Cleaning ---
#         if not self.is_local_pipeline_available:
#             return "Summarization failed: Local summarization pipeline not available."
#         if not text_to_summarize or not isinstance(text_to_summarize, str):
#              return "Input text is empty or invalid."

#         cleaned_text = clean_text_for_summarization(text_to_summarize)
#         min_meaningful_length = 50
#         if not cleaned_text or len(cleaned_text) < min_meaningful_length:
#             # ... (original short text message logic) ...
#             return "Text too short or invalid after cleaning to provide a summary."

#         # --- Determine Workflow: Direct or Chunking ---
#         needs_chunking = False
#         base_summary = "Summarization process did not complete." # Default error
#         final_summary = base_summary # Initialize final summary

#         try:
#             initial_token_count = self._get_token_count(cleaned_text)
#             print(f"Summarization Agent: Cleaned text estimated token count: {initial_token_count}")
#             if initial_token_count > self.model_max_input_tokens:
#                 needs_chunking = True
#                 print(f"Summarization Agent: Text requires chunking (tokens: {initial_token_count} > {self.model_max_input_tokens}).")
#             else:
#                  print(f"Summarization Agent: Text within limits. Using direct summarization workflow.")
#         except Exception as e:
#              print(f"Summarization Agent WARNING: Error during token count check: {e}. Assuming direct summarization.")
#              needs_chunking = False

#         # --- Execute Workflow ---
#         if not needs_chunking:
#             # === Direct Summarization Workflow ===
#             print(f"Summarization Agent: Summarizing directly locally...")
#             try:
#                 direct_start_time = time.time()
#                 summary_output = self.pipeline(
#                     cleaned_text,
#                     max_length=self.final_summary_max_len, min_length=self.final_summary_min_len,
#                     do_sample=False, truncation=True, no_repeat_ngram_size=3, early_stopping=True,
#                 )
#                 direct_end_time = time.time()
#                 summary_text = summary_output[0]['summary_text'].strip() if summary_output and summary_output[0].get('summary_text') else None
#                 if summary_text:
#                      base_summary = summary_text
#                      print(f"Summarization Agent: Direct local summary generated in {direct_end_time - direct_start_time:.2f}s.")
#                 else:
#                      base_summary = "Summarization failed (empty output from local pipeline)."
#                      print(f"Summarization Agent WARNING: {base_summary}")
#             except Exception as e:
#                 # ... (Error handling for direct summarization - same as before) ...
#                  print(f"Summarization Agent ERROR during direct summarization: {e}")
#                  err_str = str(e).lower()
#                  if "out of memory" in err_str: base_summary = "Summarization failed: Out of memory (direct)."
#                  elif "maximum sequence length" in err_str: base_summary = f"Summarization failed: Input exceeds local model max length ({self.model_max_input_tokens})."
#                  else: base_summary = f"Summarization failed (Direct: {type(e).__name__})."

#             # --- Optional LLM Refinement for Direct Summary ---
#             final_summary = base_summary # Start with the local summary
#             is_successful_base = isinstance(base_summary, str) and not any(err in base_summary.lower() for err in ["failed", "error"])
#             if self.is_llm_available and is_successful_base:
#                 print("Summarization Agent: Attempting LLM refinement for direct summary...")
#                 refined_summary = self._refine_summary_with_llm(base_summary)
#                 if refined_summary:
#                     final_summary = refined_summary
#                     print("Summarization Agent: Using LLM refined summary.")
#                 else:
#                     print("Summarization Agent INFO: LLM refinement failed or empty. Using original direct summary.")
#             elif not is_successful_base:
#                  print("Summarization Agent INFO: Skipping LLM refinement due to error in direct summarization.")
#             else: # LLM not available
#                 print("Summarization Agent INFO: LLM refinement not available. Using direct local summary.")

#         else:
#             # === Chunked Summarization Workflow ===
#             print("Summarization Agent: Starting chunked summarization workflow...")
#             chunking_start_time = time.time()

#             # 1. Chunk Text
#             text_chunks = self._chunk_text_by_tokens(cleaned_text)
#             if not text_chunks:
#                  base_summary = "Summarization failed: Text could not be effectively chunked."
#             else:
#                 # 2. Summarize Chunks Locally
#                 chunk_summaries = []
#                 total_chunks = len(text_chunks)
#                 for i, chunk in enumerate(text_chunks):
#                     chunk_summary = self._summarize_single_chunk(chunk, i, total_chunks)
#                     if chunk_summary: chunk_summaries.append(chunk_summary)
#                     # time.sleep(0.05) # Optional small delay

#                 if not chunk_summaries:
#                      base_summary = "Summarization failed: No chunks could be summarized successfully."
#                 else:
#                      print(f"Summarization Agent: Successfully summarized {len(chunk_summaries)}/{total_chunks} chunks locally.")
#                      combined_summaries_text = "\n\n".join(chunk_summaries)
#                      combined_tokens = self._get_token_count(combined_summaries_text)
#                      print(f"Summarization Agent: Combined intermediate summaries (length: {len(combined_summaries_text)} chars, {combined_tokens} tokens).")

#                      # 3. Final Synthesis Step (LLM Preferred, Local Fallback)
#                      if self.is_llm_available:
#                          print("Summarization Agent: Attempting final synthesis using Together AI LLM...")
#                          llm_final_summary = self._synthesize_summary_with_llm(combined_summaries_text)
#                          if llm_final_summary:
#                              base_summary = llm_final_summary
#                              print("Summarization Agent: Final summary synthesized by LLM.")
#                          else:
#                              print("Summarization Agent WARNING: LLM synthesis failed or returned empty. Falling back to local summarization for final step.")
#                              base_summary = self._summarize_combined_locally(combined_summaries_text)
#                      else:
#                          print("Summarization Agent INFO: LLM not available. Using local pipeline for final synthesis.")
#                          base_summary = self._summarize_combined_locally(combined_summaries_text)

#             # For chunked workflow, the base_summary IS the final_summary (no separate refinement step after synthesis)
#             final_summary = base_summary
#             total_chunking_time = time.time() - chunking_start_time
#             print(f"Summarization Agent: Chunking workflow completed in {total_chunking_time:.2f}s.")


#         # --- Final Output and Logging ---
#         # Optional: Add final noise check here if desired, applied to `final_summary`
#         if isinstance(final_summary, str) and re.search(r'(?:#\s*\d+\s*){3,}', final_summary):
#             print(f"WARNING: Final summary may contain noise patterns: '{final_summary[:100]}...'")

#         end_overall_time = time.time()
#         print(f"Summarization Agent: Total processing time: {end_overall_time - start_overall_time:.2f} seconds.")

#         return final_summary



import time
import traceback
from typing import Optional, TypedDict

# LangChain imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    GEMINI_AVAILABLE = True
except ImportError:
    print("WARNING: 'langchain-google-genai' not found. Install with: pip install langchain-google-genai")
    ChatGoogleGenerativeAI = None
    PromptTemplate = None
    GEMINI_AVAILABLE = False

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    LANGRAPH_AVAILABLE = True
except ImportError:
    print("WARNING: 'langgraph' not found. Install with: pip install langgraph")
    StateGraph = None
    END = None
    LANGRAPH_AVAILABLE = False

from utils import clean_text_for_summarization

# Configuration
try:
    from config import GEMINI_API_KEY, GEMINI_MODEL_NAME_FOR_SUMMARY as DEFAULT_GEMINI_SUMMARY_MODEL
except ImportError:
    print("WARNING: Config not found. Using defaults.")
    GEMINI_API_KEY = "AIzaSyACQwN6IEzGeB59hUvVhGwpWJQiaHr5q9k"
    DEFAULT_GEMINI_SUMMARY_MODEL = "gemini-2.0-flash-lite"

class SummarizationState(TypedDict):
    original_text: str
    cleaned_text: Optional[str]
    summary: Optional[str]
    error_message: Optional[str]
    retry_count: int

class SummarizationAgent:
    def __init__(self):
        # Summary length parameters
        self.final_summary_max_len_words = 250
        self.final_summary_min_len_words = 50
        self.gemini_max_output_tokens = int(self.final_summary_max_len_words * 2.5) + 150

        # LLM and workflow setup
        self.llm = None
        self.chain = None
        self.app = None
        self.is_llm_available = False
        self.llm_model_name = DEFAULT_GEMINI_SUMMARY_MODEL

        if GEMINI_AVAILABLE and LANGRAPH_AVAILABLE and GEMINI_API_KEY:
            try:
                # Initialize LangChain LLM
                generation_config = {"max_output_tokens": self.gemini_max_output_tokens}
                self.llm = ChatGoogleGenerativeAI(
                    model=self.llm_model_name,
                    google_api_key=GEMINI_API_KEY,
                    temperature=0.4,
                    generation_config=generation_config,
                )
                # Define prompt template
                self.summarization_prompt = PromptTemplate(
                    input_variables=["text", "min_words", "max_words"],
                    template="""
                    Please provide a concise and comprehensive summary of the following text.
                    The summary should capture the main points, key arguments, and significant findings.
                    Aim for a summary length between {min_words} and {max_words} words.
                    Ensure the summary is well-written, coherent, and easy to understand.
                    Do not add any information that is not present in the original text.
                    Do not start the summary with phrases like "The text discusses", "This document is about", or "In summary". Begin directly with the summary content.
                    Output ONLY the summary text.

                    --- TEXT TO SUMMARIZE ---
                    {text}
                    --- END TEXT TO SUMMARIZE ---

                    SUMMARY:
                    """
                )
                self.chain = self.summarization_prompt | self.llm
                self.is_llm_available = True

                # Setup LangGraph workflow
                workflow = StateGraph(SummarizationState)
                workflow.add_node("clean_text", self.clean_text_node)
                workflow.add_node("summarize", self.summarize_node)
                workflow.add_node("error", self.error_node)
                workflow.set_entry_point("clean_text")
                workflow.add_edge("clean_text", "summarize")
                workflow.add_conditional_edges("summarize", self.should_retry)
                workflow.add_edge("error", END)
                self.app = workflow.compile()

                print(f"Summarization Agent: Initialized with LangChain and LangGraph.")
            except Exception as e:
                print(f"Error initializing agent: {e}")
                traceback.print_exc()
        else:
            print("Summarization Agent: Required dependencies or API key missing.")

        # Status report
        print(f"\n--- Summarization Agent Status ---")
        print(f"LangChain Integration: {'Enabled' if GEMINI_AVAILABLE else 'Disabled'}")
        print(f"LangGraph Integration: {'Enabled' if LANGRAPH_AVAILABLE else 'Disabled'}")
        print(f"Gemini LLM Available: {self.is_llm_available}")
        if self.is_llm_available:
            print(f"--> Model: {self.llm_model_name}")
            print(f"--> Target Summary Length: {self.final_summary_min_len_words}-{self.final_summary_max_len_words} words")
            print(f"--> Max Output Tokens: {self.gemini_max_output_tokens}")
        print("-" * 34)

    def clean_text_node(self, state: SummarizationState) -> SummarizationState:
        cleaned_text = clean_text_for_summarization(state["original_text"])
        return {"cleaned_text": cleaned_text}

    def summarize_node(self, state: SummarizationState) -> SummarizationState:
        try:
            response = self.chain.invoke({
                "text": state["cleaned_text"],
                "min_words": self.final_summary_min_len_words,
                "max_words": self.final_summary_max_len_words
            })
            summary = response.content.strip()
            print(f"Summarization Agent: Successfully generated summary.")
            return {"summary": summary}
        except Exception as e:
            print(f"Summarization Agent: Error in summarization - {e}")
            return {"retry_count": state["retry_count"] + 1}

    def error_node(self, state: SummarizationState) -> SummarizationState:
        return {"error_message": "Failed to summarize after multiple attempts."}

    def should_retry(self, state: SummarizationState) -> str:
        if state["summary"] is not None:
            return END
        elif state["retry_count"] < 3:
            return "summarize"
        else:
            return "error"

    def summarize(self, text_to_summarize: str) -> str:
        start_overall_time = time.time()

        if not self.is_llm_available or not self.app:
            return "Summarization failed: Service not available."
        if not text_to_summarize or not isinstance(text_to_summarize, str):
            return "Input text is empty or invalid."
        if len(text_to_summarize) < 50:
            return "Text too short for meaningful summary."

        initial_state = {
            "original_text": text_to_summarize,
            "cleaned_text": None,
            "summary": None,
            "error_message": None,
            "retry_count": 0
        }
        print(f"Summarization Agent: Starting summarization process...")
        final_state = self.app.invoke(initial_state)

        end_overall_time = time.time()
        print(f"Summarization Agent: Total processing time: {end_overall_time - start_overall_time:.2f} seconds")

        return final_state["summary"] if final_state["summary"] else final_state["error_message"] or "Unknown error."