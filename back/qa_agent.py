# # qa_agent.py

# import time
# import traceback
# import torch
# from typing import List, Tuple, Optional, Dict, Any

# # ChromaDB Imports
# import chromadb
# from chromadb.api.models.Collection import Collection

# # Existing Imports
# from sentence_transformers import SentenceTransformer, util
# from sentence_transformers.cross_encoder import CrossEncoder
# from transformers import Pipeline
# from utils import clean_text_for_summarization # Assuming this util exists
# from config import (
#     MIN_TEXT_LENGTH, DEFAULT_TOP_K_RETRIEVAL, DEFAULT_TOP_K_RERANK,
#     DEFAULT_RERANK_THRESHOLD, DEFAULT_QA_CONFIDENCE_THRESHOLD,
#     TOGETHER_API_KEY, DEFAULT_GENERATIVE_AI_MODEL_NAME # Ensure these are in config.py
# )

# # Together.ai Import
# try:
#     from together import Together
#     TOGETHER_AVAILABLE = True
# except ImportError:
#     print("WARNING: 'together' library not found. LLM generation will be unavailable.")
#     print("Install with: pip install together")
#     Together = None
#     TOGETHER_AVAILABLE = False

# class QuestionAnsweringAgent:
#     """
#     Agent responsible for answering questions based on article context.
#     Uses ChromaDB for retrieval, cross-encoder for re-ranking, and the Together.ai API
#     for generating answers.
#     """
#     def __init__(self,
#                  embedding_model: Optional[SentenceTransformer],
#                  cross_encoder_model: Optional[CrossEncoder],
#                  qa_pipeline: Optional[Pipeline], # Kept for potential future use/fallback
#                  chroma_collection: Optional[Collection],
#                  embedding_device: str,
#                  cross_encoder_device: Optional[str] = None,
#                  top_k_retrieval: int = DEFAULT_TOP_K_RETRIEVAL,
#                  top_k_rerank: int = DEFAULT_TOP_K_RERANK,
#                  rerank_threshold: float = DEFAULT_RERANK_THRESHOLD,
#                  qa_confidence_threshold: float = DEFAULT_QA_CONFIDENCE_THRESHOLD,
#                  max_context_chunks: int = 5
#                  ):
#         """
#         Initializes the QuestionAnsweringAgent.
#         (Parameters description omitted for brevity - same as before)
#         """
#         self.embedding_model = embedding_model
#         self.cross_encoder_model = cross_encoder_model
#         self.qa_pipeline = qa_pipeline # Local QA model (currently not primary path)
#         self.chroma_collection = chroma_collection
#         self.embedding_device = embedding_device
#         self.cross_encoder_device = cross_encoder_device if cross_encoder_device else embedding_device
#         self.top_k_retrieval = top_k_retrieval
#         self.top_k_rerank = top_k_rerank
#         self.rerank_threshold = rerank_threshold
#         self.qa_confidence_threshold = qa_confidence_threshold # For local QA if used
#         self.max_context_chunks = max_context_chunks

#         # Initialize Together.ai client
#         self.together_client = None
#         self.is_llm_available = False
#         if TOGETHER_AVAILABLE and Together:
#             if not TOGETHER_API_KEY or TOGETHER_API_KEY == "":
#                  print("WARNING: TOGETHER_API_KEY not set in config.py or environment. LLM generation unavailable.")
#             else:
#                 try:
#                     self.together_client = Together(api_key=TOGETHER_API_KEY)
#                     # Optional: Add a test call here if needed, e.g., list models
#                     # self.together_client.models.list()
#                     self.is_llm_available = True
#                     print("Together AI client initialized successfully.")
#                 except Exception as e:
#                     print(f"ERROR initializing Together AI client: {e}")
#                     traceback.print_exc()
#                     self.together_client = None # Ensure it's None on failure
#         else:
#              print("Together AI library not available. LLM generation disabled.")


#         # Availability Checks (Individual Components)
#         self.is_embedding_available = embedding_model is not None
#         self.is_cross_encoder_available = cross_encoder_model is not None
#         self.is_qa_pipeline_available = qa_pipeline is not None # Local QA pipeline status
#         self.is_chroma_available = chroma_collection is not None

#         # Internal state
#         self.current_article_title: Optional[str] = None
#         self.last_considered_sources: List[str] = []
#         self.last_considered_sources_before_threshold: List[str] = []

#         print("\n--- QA Agent Configuration ---")
#         print(f"Embedding Model Available: {self.is_embedding_available} (on {self.embedding_device})")
#         print(f"Cross-Encoder Model Available: {self.is_cross_encoder_available} (on {self.cross_encoder_device})")
#         print(f"Local QA Pipeline Available: {self.is_qa_pipeline_available}") # Status of local pipeline
#         print(f"ChromaDB Collection Available: {self.is_chroma_available}")
#         if self.is_chroma_available:
#             print(f"Chroma Collection Name: '{self.chroma_collection.name}'")
#         print(f"Together AI LLM Available: {self.is_llm_available}") # Status of LLM client
#         if self.is_llm_available:
#              print(f"Together AI Model: '{DEFAULT_GENERATIVE_AI_MODEL_NAME}'")
#         print("-" * 30)
#         print(f"Top K Retrieval: {self.top_k_retrieval}")
#         print(f"Top K Re-rank: {self.top_k_rerank}")
#         print(f"Re-rank Threshold: {self.rerank_threshold}")
#         print(f"Max Context Chunks for LLM: {self.max_context_chunks}")
#         # print(f"Local QA Confidence Threshold: {self.qa_confidence_threshold}") # Less relevant now
#         print("-" * 30)

#     # --- NEW: is_ready property ---
#     @property
#     def is_ready(self) -> bool:
#         """Checks if the agent has the essential components loaded to answer questions."""
#         # Essential components for the current RAG+LLM workflow:
#         # Embedding model, ChromaDB, and the LLM client.
#         # Cross-encoder is beneficial but has a fallback. Local QA pipeline isn't used in main path.
#         return (
#             self.is_embedding_available and
#             self.is_chroma_available and
#             self.is_llm_available # Check if the Together client initialized
#         )

#     # --- NEW: get_missing_components_message method ---
#     def get_missing_components_message(self) -> str:
#         """Returns a string listing the essential missing components."""
#         if self.is_ready:
#             return "All essential components available."

#         missing = []
#         if not self.is_embedding_available:
#             missing.append("Embedding Model")
#         if not self.is_chroma_available:
#             missing.append("Vector Database (ChromaDB)")
#         if not self.is_llm_available:
#             missing.append("LLM Client (Together AI)")
#         # Optional: Add non-essential but useful components
#         # if not self.is_cross_encoder_available:
#         #     missing.append("Cross-Encoder Model (Re-ranking)")

#         if not missing: # Should not happen if is_ready is False, but safeguard
#             return "Agent not ready for unknown reasons."

#         return f"Missing: {', '.join(missing)}"


#     def _retrieve_top_chunks(self, question: str, context_db_id: int) -> List[Dict[str, Any]]:
#         """
#         Retrieves relevant text chunks from ChromaDB based on the question and article ID.
#         (Implementation remains the same as your provided code)
#         """
#         if not self.is_embedding_available or not self.is_chroma_available:
#             print("Error: Retrieval system unavailable (Embedding or ChromaDB missing).")
#             return []

#         try:
#             # Ensure embedding model is on the correct device before encoding
#             # (Usually handled by SentenceTransformer itself if initialized correctly)
#             question_embedding = self.embedding_model.encode(
#                 question,
#                 convert_to_tensor=False, # Chroma typically doesn't need tensors
#                 device=self.embedding_device # Explicitly specify device
#             ).tolist() # Convert to list for JSON compatibility if needed

#             print(f"Querying ChromaDB collection '{self.chroma_collection.name}' with {self.top_k_retrieval} results for sql_article_id='{context_db_id}'")
#             results = self.chroma_collection.query(
#                 query_embeddings=[question_embedding],
#                 n_results=self.top_k_retrieval,
#                 where={"sql_article_id": str(context_db_id)}, # Ensure ID is string
#                 include=['metadatas', 'documents', 'distances'] # Ensure 'documents' is included
#             )

#             retrieved_chunks = []
#             # Check if results are valid and contain expected lists
#             docs = results.get('documents', [[]])[0]
#             metas = results.get('metadatas', [[]])[0]
#             dists = results.get('distances', [[]])[0]

#             if not docs: # Handle case where query returns no documents
#                  print("ChromaDB query returned no documents for this article ID and question.")
#                  return []

#             # Ensure all lists have the same length
#             min_len = min(len(docs), len(metas), len(dists))
#             if min_len < len(docs) or min_len < len(metas) or min_len < len(dists):
#                 print("Warning: Mismatch in lengths of documents, metadatas, or distances from ChromaDB query.")

#             for i in range(min_len):
#                 doc = docs[i]
#                 meta = metas[i]
#                 dist = dists[i]
#                 if doc and meta: # Basic check for non-empty content
#                     # Cosine distance to similarity score (assuming space="cosine")
#                     # If space="l2", distance is Euclidean, lower is better.
#                     # Check your ChromaDB setup if scores seem inverted.
#                     similarity_score = 1.0 - dist if dist is not None else 0.0
#                     retrieved_chunks.append({
#                         'text': doc,
#                         'metadata': meta,
#                         'bi_score': similarity_score, # Store initial retrieval score
#                         # 'rank_score' will be added/updated after reranking
#                     })
#                 else:
#                      print(f"Warning: Skipping invalid chunk at index {i} from ChromaDB result (empty doc or metadata).")


#             # Sort by initial retrieval score (higher is better for similarity)
#             retrieved_chunks.sort(key=lambda x: x.get('bi_score', 0.0), reverse=True)
#             print(f"Retrieved {len(retrieved_chunks)} valid chunks from ChromaDB.")
#             return retrieved_chunks

#         except Exception as e:
#             print(f"ERROR during ChromaDB query: {e}")
#             traceback.print_exc()
#             return []


#     def _rerank_sections(self, question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """
#         Re-ranks retrieved chunks using the cross-encoder.
#         (Implementation remains the same as your provided code)
#         """
#         if not chunks:
#             print("No chunks provided for reranking.")
#             return []

#         if not self.is_cross_encoder_available:
#             print("Warning: Cross-encoder unavailable. Using bi-encoder scores for ranking.")
#             # Sort by bi_score if cross-encoder isn't there
#             chunks.sort(key=lambda x: x.get('bi_score', 0.0), reverse=True)
#             # Assign bi_score as rank_score for consistency
#             for chunk in chunks:
#                 chunk['rank_score'] = chunk.get('bi_score', 0.0)
#             # Return top K based on bi-encoder score
#             return chunks[:self.top_k_rerank] # Return up to top_k_rerank items

#         try:
#             # Prepare pairs for cross-encoder: (question, chunk_text)
#             # Consider only the top N chunks based on bi_score for reranking efficiency
#             # Already sorted by bi_score in _retrieve_top_chunks
#             chunks_to_rerank = chunks[:self.top_k_rerank] # Take top K for reranking input
#             if not chunks_to_rerank:
#                  return []

#             question_chunk_pairs = [(question, chunk['text']) for chunk in chunks_to_rerank]

#             print(f"Reranking top {len(question_chunk_pairs)} chunks using cross-encoder...")
#             scores = self.cross_encoder_model.predict(
#                 question_chunk_pairs,
#                 # apply_softmax=False, # Usually False for reranking scores
#                 # convert_to_tensor=True, # predict often returns numpy array directly
#                 batch_size=32, # Adjust based on VRAM
#                 show_progress_bar=True, # Show progress for longer reranking
#                 # device=self.cross_encoder_device # Specify device if predict supports it
#             )

#             # Add the cross-encoder score to each chunk dictionary
#             for i, score in enumerate(scores):
#                 # Ensure score is a standard float
#                 chunks_to_rerank[i]['rank_score'] = float(score)

#             # Sort the chunks based on the new cross-encoder rank_score (higher is better)
#             chunks_to_rerank.sort(key=lambda x: x['rank_score'], reverse=True)
#             print(f"Re-ranking complete.")
#             # Log top few scores for debugging
#             # for i, chunk in enumerate(chunks_to_rerank[:3]):
#             #     print(f"  Rank {i+1}: Score={chunk['rank_score']:.4f}, Orig Section='{chunk.get('metadata',{}).get('original_section','?')}'")

#             return chunks_to_rerank

#         except Exception as e:
#             print(f"ERROR during cross-encoder re-ranking: {e}")
#             traceback.print_exc()
#             # Fallback: return chunks sorted by bi_score if reranking fails
#             print("Falling back to bi-encoder scores due to reranking error.")
#             chunks.sort(key=lambda x: x.get('bi_score', 0.0), reverse=True)
#             for chunk in chunks:
#                 chunk['rank_score'] = chunk.get('bi_score', 0.0) # Ensure rank_score exists
#             return chunks[:self.top_k_rerank] # Return top K


#     def _extract_answer_local(self, question: str, candidates: List[Dict[str, Any]]) -> Tuple[Optional[str], float, List[str]]:
#         """
#         (Kept for potential fallback - Implementation remains the same)
#         Extracts an answer using the local QA pipeline.
#         """
#         if not self.is_qa_pipeline_available or not candidates:
#             print("Local QA pipeline not available or no candidate chunks.")
#             return None, 0.0, []

#         # Combine candidate texts into a single context string
#         context_parts = []
#         context_sources = []
#         for candidate in candidates:
#             text = candidate.get('text', '')
#             metadata = candidate.get('metadata', {})
#             if text:
#                 context_parts.append(text.strip()) # Add cleaned text
#                 # Create source description
#                 source_desc = f"Chunk {metadata.get('chunk_index', '?')}"
#                 orig_section = metadata.get('original_section')
#                 if orig_section and orig_section != "Unknown":
#                     source_desc += f" (from section: '{orig_section}')"
#                 context_sources.append(source_desc)

#         if not context_parts:
#             print("No valid text found in candidate chunks for local QA.")
#             return None, 0.0, []

#         # Join context parts with a clear separator
#         context_string = "\n\n ***---*** \n\n".join(context_parts)

#         print(f"Running local QA pipeline with context length: {len(context_string)}")
#         try:
#             # Run the pipeline
#             result = self.qa_pipeline(
#                 question=question,
#                 context=context_string,
#                 top_k=1, # Get only the best answer
#                 handle_impossible_answer=True, # Important for no-answer cases
#                 max_answer_len=150 # Limit answer length
#             )

#             # Process the result (can be dict or list)
#             best_answer_info = result[0] if isinstance(result, list) else result

#             answer_text = best_answer_info.get('answer')
#             confidence = best_answer_info.get('score', 0.0)

#             # Handle potentially empty answers or low confidence
#             if not answer_text or answer_text.strip() == "":
#                  print("Local QA pipeline returned an empty answer string.")
#                  return None, confidence, context_sources

#             if confidence < self.qa_confidence_threshold:
#                 print(f"Local QA answer confidence ({confidence:.4f}) below threshold ({self.qa_confidence_threshold}). Answer discarded.")
#                 return None, confidence, context_sources
#             else:
#                  print(f"Local QA Answer: '{answer_text}', Confidence: {confidence:.4f}")
#                  return answer_text, confidence, context_sources

#         except Exception as e:
#             print(f"ERROR in local QA pipeline execution: {e}")
#             traceback.print_exc()
#             return None, 0.0, context_sources


#     def _generate_answer_with_llm(self, prompt: str) -> Optional[str]:
#         """
#         Generates an answer using the Together.ai API.
#         (Implementation remains the same as your provided code)
#         """
#         if not self.is_llm_available:
#              print("Error: Together AI client not available for LLM generation.")
#              return None

#         try:
#             print(f"Calling Together AI completions API with model: {DEFAULT_GENERATIVE_AI_MODEL_NAME}")
#             # Use the 'completions.create' method for standard completion models
#             # For chat models, use 'chat.completions.create' with a messages list
#             # Assuming DEFAULT_GENERATIVE_AI_MODEL_NAME is a completion model for this structure:
#             response = self.together_client.completions.create(
#                 prompt=prompt,                  # Pass the full prompt string
#                 model=DEFAULT_GENERATIVE_AI_MODEL_NAME, # Specify the model
#                 max_tokens=250,                 # Max tokens for the generated answer (adjust)
#                 temperature=0.7,                # Controls randomness (adjust)
#                 # top_p=0.7,                    # Optional nucleus sampling
#                 # top_k=50,                     # Optional top-k sampling
#                 # repetition_penalty=1.1,       # Optional penalty for repetition
#                 stop=["\n\nHuman:", "</s>", "<|endoftext|>"] # Stop sequences
#             )

#             # Access the generated text correctly from the response object
#             if response and hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
#                  # Check if the choice object has 'text' attribute
#                  if hasattr(response.choices[0], 'text'):
#                      generated_text = response.choices[0].text.strip()
#                      print(f"Together AI Response received: '{generated_text[:150]}...'")
#                      # Post-process: Remove any trailing stop sequences if included
#                      for stop_seq in ["\n\nHuman:", "</s>", "<|endoftext|>"]:
#                           if generated_text.endswith(stop_seq):
#                                generated_text = generated_text[:-len(stop_seq)].strip()
#                      return generated_text
#                  else:
#                       print("Warning: Together AI response choice object missing 'text' attribute.")
#                       print(f"Response choice object: {response.choices[0]}")
#                       return None
#             else:
#                  print("Warning: Together AI response structure unexpected or empty choices list.")
#                  print(f"Full API Response: {response}")
#                  return None

#         except Exception as e:
#             print(f"ERROR calling Together.ai completions API: {e}")
#             # Try to print more details from the error if available
#             if hasattr(e, 'response'): # Check if it's an HTTP error from the client library
#                  try:
#                       print(f"API Response Status: {e.response.status_code}")
#                       print(f"API Response Body: {e.response.text}")
#                  except Exception:
#                       pass # Ignore errors during error reporting
#             traceback.print_exc()
#             return None


#     def _format_response(self, llm_answer: str, context_sources: List[str]) -> Dict[str, str]:
#         """
#         Formats the LLM-generated response.
#         (Implementation remains the same as your provided code)
#         """
#         # Check for explicit refusal phrases
#         refusal_phrases = [
#              "i cannot answer",
#              "i cannot provide an answer",
#              "based on the provided context",
#              "context does not contain",
#              "information is not available",
#              "unable to answer",
#         ]
#         answer_lower = llm_answer.lower()
#         is_refusal = any(phrase in answer_lower for phrase in refusal_phrases)

#         if is_refusal and context_sources:
#             # Append source info to refusals for clarity
#             sources_info = "; ".join(context_sources[:self.max_context_chunks]) # Show sources used
#             if len(context_sources) > self.max_context_chunks:
#                  sources_info += f"... (from top {self.max_context_chunks} relevant parts)"

#             # Try to make the refusal message more informative
#             formatted_response = (
#                 f"{llm_answer.strip()} "
#                 f"I reviewed the relevant parts of the document (including: {sources_info}) to arrive at this conclusion."
#             )
#             return {"response": formatted_response}
#         else:
#              # Return the LLM answer as is if it's not a refusal
#              return {"response": llm_answer.strip()}

#     # --- Main answer method ---
#     # (Your existing answer method implementation seems correct and uses the helpers)
#     def answer(self, question: str, article_title: str, context_db_id: int) -> Dict[str, Any]:
#         """
#         Answers a question using Retrieve-Rank-Generate pipeline with LLM.

#         :param question: The user's question.
#         :param article_title: Title of the article.
#         :param context_db_id: The SQL ID of the article.
#         :return: Dictionary with the response, sources or error message.
#         """
#         self.current_article_title = article_title
#         self.last_considered_sources = []
#         self.last_considered_sources_before_threshold = []
#         start_time = time.time()

#         # --- Initial Checks ---
#         if not self.is_ready: # Use the new property
#             return {"error": f"The Question Answering service is not fully available. {self.get_missing_components_message()}", "status_code": 503}
#         if not question or not question.strip():
#             return {"error": "Invalid or empty question provided.", "status_code": 400}

#         print(f"\n--- Answering question for article ID {context_db_id}: '{question[:100]}...' ---")

#         # --- 1. Retrieve ---
#         retrieved_chunks = self._retrieve_top_chunks(question, context_db_id)
#         if not retrieved_chunks:
#             return {"response": f"I couldn't find any text related to your question within the document (ID: {context_db_id})."}

#         # --- 2. Re-rank ---
#         ranked_chunks = self._rerank_sections(question, retrieved_chunks)
#         if not ranked_chunks:
#             # Should ideally not happen if retrieved_chunks is not empty, but handle anyway
#             return {"response": "An issue occurred while ranking relevant document parts. Cannot provide an answer."}

#         # Store sources before threshold filtering (for context in refusals)
#         for chunk in ranked_chunks:
#             metadata = chunk.get('metadata', {})
#             source_desc = f"Chunk {metadata.get('chunk_index', '?')}"
#             orig_section = metadata.get('original_section')
#             if orig_section and orig_section != "Unknown":
#                 source_desc += f" ('{orig_section}')"
#             self.last_considered_sources_before_threshold.append(source_desc)

#         # --- 3. Filter & Select Context ---
#         print(f"Filtering top {len(ranked_chunks)} ranked chunks with threshold {self.rerank_threshold}...")
#         final_candidates = [
#             chunk for chunk in ranked_chunks
#             if chunk.get('rank_score', -float('inf')) >= self.rerank_threshold # Handle missing score safely
#         ][:self.max_context_chunks] # Limit to max chunks

#         if not final_candidates:
#             considered_sources_str = "; ".join(self.last_considered_sources_before_threshold[:5]) if self.last_considered_sources_before_threshold else "None"
#             print(f"No chunks met the re-rank threshold ({self.rerank_threshold}). Top considered sources: {considered_sources_str}")
#             return {"response": "Although I found some related text, none of it was relevant enough to confidently answer your question based on the ranking threshold."}

#         # --- 4. Build Context and Source List ---
#         context_parts = []
#         context_sources = [] # Sources that *passed* the threshold
#         print(f"Building context from {len(final_candidates)} final candidate chunks:")
#         for i, chunk in enumerate(final_candidates):
#             text = chunk.get('text', '')
#             metadata = chunk.get('metadata', {})
#             rank_score = chunk.get('rank_score', -1)
#             if text:
#                 context_parts.append(text.strip())
#                 # Create source description
#                 source_desc = f"Chunk {metadata.get('chunk_index', '?')}"
#                 orig_section = metadata.get('original_section')
#                 if orig_section and orig_section != "Unknown":
#                     source_desc += f" (from section: '{orig_section}')"
#                 # Include score for debugging/potential display
#                 source_desc_with_score = f"{source_desc} [Score: {rank_score:.3f}]"
#                 context_sources.append(source_desc_with_score) # Store sources actually used
#                 print(f"  - Including: {source_desc_with_score}")
#             else:
#                  print(f"  - Warning: Skipping candidate chunk {i} due to empty text.")


#         if not context_parts:
#              print("Error: No valid text content in the final candidate chunks after filtering.")
#              return {"error": "An internal error occurred processing the document context after filtering.", "status_code": 500}

#         # Join context with clear separators
#         context_string = "\n\n---\n\n".join(context_parts)

#         # --- 5. Construct Prompt for LLM ---
#         # Use a clear instruction format
#         prompt = (
#             f"You are an AI assistant analyzing a document.\n"
#             f"Based *only* on the following extracted context, answer the user's question.\n"
#             f"Do not use any prior knowledge or information outside the provided text.\n"
#             f"If the context does not contain the answer, clearly state that the information is not found in the provided context.\n\n"
#             f"CONTEXT:\n"
#             f"-------\n"
#             f"{context_string}\n"
#             f"-------\n\n"
#             f"QUESTION: {question}\n\n"
#             f"ANSWER:"
#         )
#         # Log prompt size for debugging (optional)
#         # print(f"Prompt context length (chars): {len(context_string)}")
#         # print(f"Full prompt length (chars): {len(prompt)}") # Be careful logging full prompt

#         # --- 6. Generate Answer with LLM ---
#         llm_answer = self._generate_answer_with_llm(prompt)

#         if llm_answer is None:
#             # Error occurred during LLM call
#             return {"error": "Failed to generate an answer using the language model. The service might be unavailable or encountered an error.", "status_code": 503}

#         # --- 7. Format Final Response ---
#         # Use the sources that were actually included in the prompt
#         final_response = self._format_response(llm_answer, context_sources)

#         # Optionally add the sources to the response for the frontend
#         final_response['sources'] = context_sources

#         end_time = time.time()
#         print(f"--- Question answering process took {end_time - start_time:.2f} seconds ---")
#         return final_response


# qa_agent.py

import time
import traceback
import torch
from typing import List, Tuple, Optional, Dict, Any

# ChromaDB Imports
import chromadb
from chromadb.api.models.Collection import Collection

# LangChain Imports
from langchain.schema.runnable import RunnableLambda, RunnableSequence

# Existing Imports
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import Pipeline
from utils import clean_text_for_summarization
from config import (
    MIN_TEXT_LENGTH, DEFAULT_TOP_K_RETRIEVAL, DEFAULT_TOP_K_RERANK,
    DEFAULT_RERANK_THRESHOLD, DEFAULT_QA_CONFIDENCE_THRESHOLD,
    GEMINI_API_KEY
)

# Google Gemini Import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("WARNING: 'google-generativeai' library not found. LLM generation will be unavailable.")
    print("Install with: pip install google-generativeai")
    genai = None
    GEMINI_AVAILABLE = False

class QuestionAnsweringAgent:
    def __init__(self,
                 embedding_model: Optional[SentenceTransformer],
                 cross_encoder_model: Optional[CrossEncoder],
                 qa_pipeline: Optional[Pipeline],
                 chroma_collection: Optional[Collection],
                 embedding_device: str,
                 cross_encoder_device: Optional[str] = None,
                 top_k_retrieval: int = DEFAULT_TOP_K_RETRIEVAL,
                 top_k_rerank: int = DEFAULT_TOP_K_RERANK,
                 rerank_threshold: float = DEFAULT_RERANK_THRESHOLD,
                 qa_confidence_threshold: float = DEFAULT_QA_CONFIDENCE_THRESHOLD,
                 max_context_chunks: int = 5
                 ):
        self.embedding_model = embedding_model
        self.cross_encoder_model = cross_encoder_model
        self.qa_pipeline = qa_pipeline
        self.chroma_collection = chroma_collection
        self.embedding_device = embedding_device
        self.cross_encoder_device = cross_encoder_device if cross_encoder_device else embedding_device
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.rerank_threshold = rerank_threshold
        self.qa_confidence_threshold = qa_confidence_threshold
        self.max_context_chunks = max_context_chunks

        # Initialize Google Gemini client
        self.gemini_model = None
        self.is_llm_available = False
        if GEMINI_AVAILABLE and genai:
            if not GEMINI_API_KEY or GEMINI_API_KEY == "":
                print("WARNING: GEMINI_API_KEY not set in config.py or environment. LLM generation unavailable.")
            else:
                try:
                    genai.configure(api_key=GEMINI_API_KEY)
                    self.gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
                    self.is_llm_available = True
                    print("Google Gemini AI client initialized successfully.")
                except Exception as e:
                    print(f"ERROR initializing Google Gemini AI client: {e}")
                    traceback.print_exc()
                    self.gemini_model = None
        else:
            print("Google Generative AI library not available. LLM generation disabled.")

        # Availability Checks
        self.is_embedding_available = embedding_model is not None
        self.is_cross_encoder_available = cross_encoder_model is not None
        self.is_qa_pipeline_available = qa_pipeline is not None
        self.is_chroma_available = chroma_collection is not None

        # Internal state
        self.current_article_title: Optional[str] = None
        self.last_considered_sources: List[str] = []
        self.last_considered_sources_before_threshold: List[str] = []

        # Print configuration (unchanged)
        print("\n--- QA Agent Configuration ---")
        print(f"Embedding Model Available: {self.is_embedding_available} (on {self.embedding_device})")
        print(f"Cross-Encoder Model Available: {self.is_cross_encoder_available} (on {self.cross_encoder_device})")
        print(f"Local QA Pipeline Available: {self.is_qa_pipeline_available}")
        print(f"ChromaDB Collection Available: {self.is_chroma_available}")
        if self.is_chroma_available:
            print(f"Chroma Collection Name: '{self.chroma_collection.name}'")
        print(f"Google Gemini LLM Available: {self.is_llm_available}")
        if self.is_llm_available:
            print(f"Google Gemini Model: 'gemini-1.5-flash-latest'")
        print("-" * 30)
        print(f"Top K Retrieval: {self.top_k_retrieval}")
        print(f"Top K Re-rank: {self.top_k_rerank}")
        print(f"Re-rank Threshold: {self.rerank_threshold}")
        print(f"Max Context Chunks for LLM: {self.max_context_chunks}")
        print("-" * 30)

    @property
    def is_ready(self) -> bool:
        return (
            self.is_embedding_available and
            self.is_chroma_available and
            self.is_llm_available
        )

    def get_missing_components_message(self) -> str:
        if self.is_ready:
            return "All essential components available."
        missing = []
        if not self.is_embedding_available:
            missing.append("Embedding Model")
        if not self.is_chroma_available:
            missing.append("Vector Database (ChromaDB)")
        if not self.is_llm_available:
            missing.append("LLM Client (Google Gemini)")
        return f"Missing: {', '.join(missing)}" if missing else "Agent not ready for unknown reasons."

    def _retrieve_top_chunks(self, question: str, context_db_id: int) -> List[Dict[str, Any]]:
        # (Unchanged from original)
        if not self.is_embedding_available or not self.is_chroma_available:
            print("Error: Retrieval system unavailable (Embedding or ChromaDB missing).")
            return []
        try:
            question_embedding = self.embedding_model.encode(
                question,
                convert_to_tensor=False,
                device=self.embedding_device
            ).tolist()
            print(f"Querying ChromaDB collection '{self.chroma_collection.name}' with {self.top_k_retrieval} results for sql_article_id='{context_db_id}'")
            results = self.chroma_collection.query(
                query_embeddings=[question_embedding],
                n_results=self.top_k_retrieval,
                where={"sql_article_id": str(context_db_id)},
                include=['metadatas', 'documents', 'distances']
            )
            retrieved_chunks = []
            docs = results.get('documents', [[]])[0]
            metas = results.get('metadatas', [[]])[0]
            dists = results.get('distances', [[]])[0]
            if not docs:
                print("ChromaDB query returned no documents for this article ID and question.")
                return []
            min_len = min(len(docs), len(metas), len(dists))
            if min_len < len(docs) or min_len < len(metas) or min_len < len(dists):
                print("Warning: Mismatch in lengths of documents, metadatas, or distances from ChromaDB query.")
            for i in range(min_len):
                doc = docs[i]
                meta = metas[i]
                dist = dists[i]
                if doc and meta:
                    similarity_score = 1.0 - dist if dist is not None else 0.0
                    retrieved_chunks.append({
                        'text': doc,
                        'metadata': meta,
                        'bi_score': similarity_score,
                    })
                else:
                    print(f"Warning: Skipping invalid chunk at index {i} from ChromaDB result (empty doc or metadata).")
            retrieved_chunks.sort(key=lambda x: x.get('bi_score', 0.0), reverse=True)
            print(f"Retrieved {len(retrieved_chunks)} valid chunks from ChromaDB.")
            return retrieved_chunks
        except Exception as e:
            print(f"ERROR during ChromaDB query: {e}")
            traceback.print_exc()
            return []

    def _rerank_sections(self, question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # (Unchanged from original)
        if not chunks:
            print("No chunks provided for reranking.")
            return []
        if not self.is_cross_encoder_available:
            print("Warning: Cross-encoder unavailable. Using bi-encoder scores for ranking.")
            chunks.sort(key=lambda x: x.get('bi_score', 0.0), reverse=True)
            for chunk in chunks:
                chunk['rank_score'] = chunk.get('bi_score', 0.0)
            return chunks[:self.top_k_rerank]
        try:
            chunks_to_rerank = chunks[:self.top_k_rerank]
            if not chunks_to_rerank:
                return []
            question_chunk_pairs = [(question, chunk['text']) for chunk in chunks_to_rerank]
            print(f"Reranking top {len(question_chunk_pairs)} chunks using cross-encoder...")
            scores = self.cross_encoder_model.predict(
                question_chunk_pairs,
                batch_size=32,
                show_progress_bar=True,
            )
            for i, score in enumerate(scores):
                chunks_to_rerank[i]['rank_score'] = float(score)
            chunks_to_rerank.sort(key=lambda x: x['rank_score'], reverse=True)
            print(f"Re-ranking complete.")
            return chunks_to_rerank
        except Exception as e:
            print(f"ERROR during cross-encoder re-ranking: {e}")
            traceback.print_exc()
            print("Falling back to bi-encoder scores due to reranking error.")
            chunks.sort(key=lambda x: x.get('bi_score', 0.0), reverse=True)
            for chunk in chunks:
                chunk['rank_score'] = chunk.get('bi_score', 0.0)
            return chunks[:self.top_k_rerank]

    def _generate_answer_with_llm(self, prompt: str) -> Optional[str]:
        # (Unchanged from original)
        if not self.is_llm_available or not self.gemini_model:
            print("Error: Google Gemini AI client not available for LLM generation.")
            return None
        try:
            print(f"Calling Google Gemini API with model: 'gemini-1.5-flash-latest'")
            generation_config = genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=250,
                temperature=0.7,
                stop_sequences=["\n\nHuman:", "</s>", "<|endoftext|>", "END_OF_RESPONSE"]
            )
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            if not response.candidates:
                block_reason = "Unknown"
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    block_reason = response.prompt_feedback.block_reason.name
                print(f"Warning: Gemini API call blocked. Reason: {block_reason}")
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    print(f"Prompt Feedback: {response.prompt_feedback}")
                return f"I am unable to provide an answer due to content safety filters ({block_reason})."
            generated_text = response.text.strip()
            print(f"Google Gemini Response received: '{generated_text[:150]}...'")
            return generated_text
        except Exception as e:
            print(f"ERROR calling Google Gemini API: {e}")
            traceback.print_exc()
            return None

    def _format_response(self, llm_answer: str, context_sources: List[str]) -> Dict[str, str]:
        # (Unchanged from original)
        refusal_phrases = [
            "i cannot answer", "i can't answer",
            "i cannot provide an answer", "i can't provide an answer",
            "based on the provided context",
            "context does not contain", "context doesn't contain",
            "information is not available", "information isn't available",
            "unable to answer",
            "i am unable to provide an answer due to content safety filters"
        ]
        answer_lower = llm_answer.lower()
        is_refusal = any(phrase in answer_lower for phrase in refusal_phrases)
        if "unable to provide an answer due to content safety filters" in answer_lower:
            return {"response": llm_answer.strip()}
        if is_refusal and "based on the provided context" in answer_lower and context_sources:
            sources_info = "; ".join(context_sources[:self.max_context_chunks])
            if len(context_sources) > self.max_context_chunks:
                sources_info += f"... (from top {self.max_context_chunks} relevant parts)"
            formatted_response = (
                f"{llm_answer.strip()} "
                f"I reviewed the relevant parts of the document (including: {sources_info}) to arrive at this conclusion."
            )
            return {"response": formatted_response}
        elif is_refusal:
            return {"response": llm_answer.strip()}
        else:
            return {"response": llm_answer.strip()}

    def answer(self, question: str, article_title: str, context_db_id: int) -> Dict[str, Any]:
        """
        Answers a question using a LangChain RunnableSequence pipeline.
        """
        self.current_article_title = article_title
        self.last_considered_sources = []
        self.last_considered_sources_before_threshold = []
        start_time = time.time()

        if not self.is_ready:
            return {"error": f"The Question Answering service is not fully available. {self.get_missing_components_message()}", "status_code": 503}
        if not question or not question.strip():
            return {"error": "Invalid or empty question provided.", "status_code": 400}

        print(f"\n--- Answering question for article ID {context_db_id}: '{question[:100]}...' ---")

        # Define pipeline steps
        def retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
            state["retrieved_chunks"] = self._retrieve_top_chunks(state["question"], state["article_id"])
            if not state["retrieved_chunks"]:
                state["response"] = f"I couldn't find any text related to your question within the document (ID: {state['article_id']})."
            return state

        def rerank(state: Dict[str, Any]) -> Dict[str, Any]:
            if "response" in state:
                return state
            state["ranked_chunks"] = self._rerank_sections(state["question"], state["retrieved_chunks"])
            if not state["ranked_chunks"]:
                state["response"] = "An issue occurred while ranking relevant document parts. Cannot provide an answer."
            return state

        def filter_and_build_context(state: Dict[str, Any]) -> Dict[str, Any]:
            if "response" in state:
                return state
            ranked_chunks = state["ranked_chunks"]
            for chunk in ranked_chunks:
                metadata = chunk.get('metadata', {})
                source_desc = f"Chunk {metadata.get('chunk_index', '?')}"
                orig_section = metadata.get('original_section')
                if orig_section and orig_section != "Unknown":
                    source_desc += f" ('{orig_section}')"
                self.last_considered_sources_before_threshold.append(source_desc)
            print(f"Filtering top {len(ranked_chunks)} ranked chunks with threshold {self.rerank_threshold}...")
            final_candidates = [
                chunk for chunk in ranked_chunks
                if chunk.get('rank_score', -float('inf')) >= self.rerank_threshold
            ][:self.max_context_chunks]
            if not final_candidates:
                considered_sources_str = "; ".join(self.last_considered_sources_before_threshold[:5]) if self.last_considered_sources_before_threshold else "None"
                print(f"No chunks met the re-rank threshold ({self.rerank_threshold}). Top considered sources: {considered_sources_str}")
                state["response"] = "Although I found some related text, none of it was relevant enough to confidently answer your question based on the ranking threshold."
                return state
            context_parts = []
            context_sources = []
            print(f"Building context from {len(final_candidates)} final candidate chunks:")
            for i, chunk in enumerate(final_candidates):
                text = chunk.get('text', '')
                metadata = chunk.get('metadata', {})
                rank_score = chunk.get('rank_score', -1)
                if text:
                    context_parts.append(text.strip())
                    source_desc = f"Chunk {metadata.get('chunk_index', '?')}"
                    orig_section = metadata.get('original_section')
                    if orig_section and orig_section != "Unknown":
                        source_desc += f" (from section: '{orig_section}')"
                    source_desc_with_score = f"{source_desc} [Score: {rank_score:.3f}]"
                    context_sources.append(source_desc_with_score)
                    print(f"  - Including: {source_desc_with_score}")
                else:
                    print(f"  - Warning: Skipping candidate chunk {i} due to empty text.")
            if not context_parts:
                state["error"] = "An internal error occurred processing the document context after filtering."
                state["status_code"] = 500
                return state
            state["context"] = "\n\n---\n\n".join(context_parts)
            state["sources"] = context_sources
            self.last_considered_sources = context_sources
            return state

        def generate_prompt(state: Dict[str, Any]) -> Dict[str, Any]:
            if "response" in state or "error" in state:
                return state
            prompt = (
                f"You are an AI assistant. Your task is to answer the user's question based *only* on the provided CONTEXT.\n"
                f"Do not use any external knowledge or information outside of this CONTEXT.\n"
                f"If the CONTEXT does not contain the information needed to answer the QUESTION, you must state that the information is not found in the provided context.\n"
                f"Do not make up answers.\n\n"
                f"CONTEXT:\n"
                f"-------\n"
                f"{state['context']}\n"
                f"-------\n\n"
                f"QUESTION: {state['question']}\n\n"
                f"ANSWER:"
            )
            state["prompt"] = prompt
            return state

        def call_llm(state: Dict[str, Any]) -> Dict[str, Any]:
            if "response" in state or "error" in state:
                return state
            state["llm_answer"] = self._generate_answer_with_llm(state["prompt"])
            if state["llm_answer"] is None:
                state["error"] = "Failed to generate an answer using the language model. The service might be unavailable or encountered an error."
                state["status_code"] = 503
            return state

        def format_response(state: Dict[str, Any]) -> Dict[str, Any]:
            if "error" in state:
                return {"error": state["error"], "status_code": state["status_code"]}
            if "response" in state:
                return {"response": state["response"]}
            final_response = self._format_response(state["llm_answer"], state["sources"])
            final_response["sources"] = state["sources"]
            return final_response

        # Create the RunnableSequence
        pipeline = RunnableSequence(
            RunnableLambda(retrieve),
            RunnableLambda(rerank),
            RunnableLambda(filter_and_build_context),
            RunnableLambda(generate_prompt),
            RunnableLambda(call_llm),
            RunnableLambda(format_response)
        )

        # Execute the pipeline
        initial_state = {"question": question, "article_id": context_db_id}
        result = pipeline.invoke(initial_state)

        end_time = time.time()
        print(f"--- Question answering process took {end_time - start_time:.2f} seconds ---")
        return result

    # Other methods (_extract_answer_local) remain unchanged but omitted for brevity