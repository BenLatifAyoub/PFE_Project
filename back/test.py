import google.generativeai as genai
import fitz  # PyMuPDF (requires `pip install PyMuPDF`)
import os
import textwrap # Optional: for better text formatting
import traceback # For printing detailed error information

# --- Configuration ---
# IMPORTANT: Replace with your actual API key
API_KEY = "AIzaSyACQwN6IEzGeB59hUvVhGwpWJQiaHr5q9k" 
# IMPORTANT: Replace with the full path to your PDF file
PDF_FILE_PATH = r"C:\Users\MSI\OneDrive\Documents\Desktop\5idma\PFE\00\0a\PMC2607539\PMC2607539\pone.0004153.pdf" 

# Choose the model (e.g., 'gemini-1.5-pro-latest', 'gemini-1.0-pro')
# Gemini 1.5 Pro is generally recommended for potentially long documents.
# Using gemini-1.5-flash as it might be faster and sufficient for this task
MODEL_NAME = "gemini-1.5-flash-latest" # Or use "gemini-1.5-pro-latest" if needed

# --- Helper Function (Optional) ---
def to_markdown(text):
    """Formats plain text to Markdown for better display in compatible environments."""
    if not text: return ""
    text = text.replace('•', '  *') # Basic bullet point conversion
    return textwrap.indent(text, '> ', predicate=lambda _: True)

# --- PDF Text Extraction Function ---
def extract_text_from_pdf(pdf_path):
    """
    Extracts text content from a PDF file using PyMuPDF (fitz).
    Uses a 'with' statement for proper resource management.
    """
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found at '{pdf_path}'")
        return None
    if not pdf_path.lower().endswith('.pdf'):
         print(f"❌ Error: File '{pdf_path}' does not seem to be a PDF.")
         return None

    print(f" Reading PDF: {pdf_path}")
    try:
        full_text = ""
        page_count = 0
        with fitz.open(pdf_path) as doc:
            page_count = len(doc)
            print(f" Attempting to extract text from {page_count} pages...")
            for page_num in range(page_count):
                page = doc.load_page(page_num)
                page_text = page.get_text("text", sort=True) # sort=True helps with reading order
                full_text += page_text
                # Add page breaks only if needed for very long docs, can increase token count
                # full_text += f"\n--- End of Page {page_num + 1} ---\n" 

        print(f" Successfully extracted text from {page_count} pages.")

        # Basic cleanup: remove excessive blank lines and join lines intelligently
        lines = full_text.splitlines()
        cleaned_lines = [line for line in lines if line.strip()] # Keep only non-empty lines
        # Simple rejoining logic (may need adjustment for complex layouts)
        cleaned_text = ""
        for i, line in enumerate(cleaned_lines):
            cleaned_text += line
            # Add a space if the line likely doesn't end with punctuation, unless it's the last line
            if i < len(cleaned_lines) - 1 and not line.strip().endswith(('.', ':', ';', '!', '?')):
                 cleaned_text += " " 
            else:
                 cleaned_text += "\n" # Add newline after lines ending in punctuation or the last line

        full_text = cleaned_text.strip() # Remove leading/trailing whitespace

        if not full_text:
             print("⚠️ Warning: Extracted text is empty. The PDF might contain images or have complex formatting.")

        return full_text
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
        traceback.print_exc()
        return None

# --- AI Generation Function: Course Outline ---
def generate_course_outline(api_key, model_name, article_text):
    """
    Generates a structured course outline based on the provided article text.
    """
    if not api_key or api_key == "YOUR_API_KEY": # Check if placeholder is still there
        print("❌ Error: Please replace 'YOUR_API_KEY' with your actual API key.")
        return None
    if not article_text:
         print("❌ Error: No article text provided to generate the course outline from.")
         return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        # --- Prompt for Course Generation ---
        course_prompt = f"""
        Act as an expert instructional designer and subject matter expert in the field related to the article.
        Analyze the following scientific article text meticulously. Based **strictly and solely** on the information presented in this text, create a comprehensive, well-structured, and easy-to-understand course outline.

        The course outline MUST include:
        1.  **Course Title:** A concise and informative title reflecting the article's core topic.
        2.  **Target Audience:** A specific description of who would benefit most from this course (e.g., "Advanced undergraduate neuroscience students," "Clinical researchers investigating X," "Bioinformaticians working with Y data").
        3.  **Course Goal:** A single sentence describing the overall purpose of the course.
        4.  **Learning Objectives:** A list of 5-7 specific, measurable, achievable, relevant, and time-bound (SMART-style, if possible) objectives. Use action verbs (e.g., "Explain...", "Compare...", "Analyze...", "Interpret...", "Apply...", "Summarize..."). These objectives must cover the key concepts, methods, findings, and conclusions presented *in the article*.
        5.  **Course Modules:** Break down the article's content into logical modules. Follow the typical structure of a scientific paper (e.g., Introduction/Background, Materials & Methods, Results [potentially split by experiment/figure], Discussion, Conclusion/Limitations).
            *   For **each** module:
                *   Assign a clear, descriptive title (e.g., "Module 1: Introduction and Background to [Topic]").
                *   Provide a detailed course (at least 4-6 sentences) covering the **key information, concepts, procedures, data, or arguments** presented in the corresponding section of the article. Ensure all significant details from that section are captured accurately. Use clear and simple language where possible, while retaining scientific accuracy.
                *   List 2-3 specific learning points or key takeaways for that module, directly derived from the article content covered.
        6.  **Key Terminology:** List 8-10 essential terms found within the article. Provide concise definitions for each term based *only* on how they are used or implicitly defined within the article text.
        7.  **Overall Course Summary:** A concluding paragraph (4-5 sentences) that synthesizes the main points of the article and reiterates the core message or findings covered in the course.

        **Formatting Instructions:**
        *   Use Markdown for structure (e.g., `## Course Title`, `### Module 1: Title`, `* Bullet points`).
        *   Ensure the language is accessible to the defined target audience.
        *   Adhere strictly to the content provided in the article text below. Do not add external information or interpretations.

        **Scientific Article Text:**
        --- START OF ARTICLE TEXT ---
        {article_text}
        --- END OF ARTICLE TEXT ---

        Generate the detailed course outline now.
        """

        # --- Configuration for Generation ---
        generation_config = genai.types.GenerationConfig(
             temperature=0.4 # Slightly more deterministic for factual structuring
             # max_output_tokens=8192 # Adjust if needed, but Flash/Pro 1.5 handle large contexts well
        )
        # Adjust safety settings if needed, defaults are usually fine
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        print(f"\n Sending request to {model_name} to generate Course Outline...")
        response = model.generate_content(
            course_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
            request_options={'timeout': 600} # Increase timeout for potentially long generation
        )

        # --- Process Response ---
        course_text = None
        try:
            course_text = response.text
            print("✅ Course Outline generation successful.")
        except ValueError: # Handle cases where .text might be empty due to safety or other issues
            try:
                 if response.candidates:
                     course_text = response.candidates[0].content.parts[0].text
                     print("✅ Course Outline generation successful (extracted from candidate).")
                 else:
                     raise ValueError("No candidates found in response.")
            except Exception as e:
                 print(f"⚠️ Warning: Could not extract text from course generation response. Error: {e}")
                 print(f"   Response Block Reason: {response.prompt_feedback.block_reason if response.prompt_feedback else 'N/A'}")
                 print(f"   Response Safety Ratings: {response.prompt_feedback.safety_ratings if response.prompt_feedback else 'N/A'}")
        except Exception as e:
             print(f"⚠️ Warning: Error processing course response: {e}")
             print(f"   Raw Course Response: {response}")

        return course_text

    except Exception as e:
        print(f"❌ An error occurred during Course Outline generation API interaction: {e}")
        traceback.print_exc()
        return None

# --- AI Generation Function: Quizzes from Course Outline ---
def generate_quizzes_from_course(api_key, model_name, course_outline_text):
    """
    Generates 20 quiz questions based *only* on the provided course outline text.
    """
    if not api_key or api_key == "YOUR_API_KEY":
        print("❌ Error: Please replace 'YOUR_API_KEY' with your actual API key.")
        return None
    if not course_outline_text:
         print("❌ Error: No course outline text provided to generate quizzes from.")
         return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        # --- Prompt for Quiz Generation ---
        quiz_prompt = f"""
        Act as an assessment specialist. You are provided with a detailed course outline derived from a scientific article.
        Your task is to create exactly 20 quiz questions based **strictly and solely** on the information presented in this course outline. Do not refer back to any original article or external knowledge.

        The quiz must adhere to the following requirements:
        1.  **Total Questions:** Exactly 20 questions.
        2.  **Source Material:** Base all questions and answers *only* on the provided course outline text below.
        3.  **Question Types:** Include a mix of question types covering different aspects of the course outline. Aim for approximately:
            *   8 Multiple Choice (MC) questions (with 4 distinct options: A, B, C, D).
            *   7 True/False (TF) questions.
            *   5 Short Answer (SA) questions (requiring a brief, factual answer based on the outline).
        4.  **Content Focus:** Questions should test understanding of the learning objectives, key concepts from modules, definitions of key terminology, and the overall summary points mentioned in the course outline.
        5.  **Clarity:** Questions should be clear, unambiguous, and directly answerable from the provided text.
        6.  **Answers:** Provide the correct answer immediately after each question.
            *   For MC: Clearly indicate the correct option (e.g., "Answer: C").
            *   For TF: Clearly state "Answer: True" or "Answer: False".
            *   For SA: Provide a concise, accurate model answer based *only* on the course outline content (e.g., "Answer: [Brief factual answer from outline]").
        7.  **Formatting:**
            *   Number each question sequentially from 1 to 20.
            *   Clearly label the question type in parentheses after the number (e.g., "1. (MC)", "9. (TF)", "16. (SA)").
            *   Format options for MC questions clearly (A, B, C, D).

        **Course Outline Text:**
        --- START OF COURSE OUTLINE TEXT ---
        {course_outline_text}
        --- END OF COURSE OUTLINE TEXT ---

        Generate the 20 quiz questions with answers now.
        """

        # --- Configuration for Generation ---
        generation_config = genai.types.GenerationConfig(
             temperature=0.5 # Balance creativity for questions with factual accuracy
        )
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        print(f"\n Sending request to {model_name} to generate 20 Quizzes...")
        response = model.generate_content(
            quiz_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
            request_options={'timeout': 600} # Increase timeout
        )

        # --- Process Response ---
        quiz_text = None
        try:
            quiz_text = response.text
            print("✅ Quiz generation successful.")
        except ValueError:
            try:
                 if response.candidates:
                     quiz_text = response.candidates[0].content.parts[0].text
                     print("✅ Quiz generation successful (extracted from candidate).")
                 else:
                     raise ValueError("No candidates found in response.")
            except Exception as e:
                 print(f"⚠️ Warning: Could not extract text from quiz generation response. Error: {e}")
                 print(f"   Response Block Reason: {response.prompt_feedback.block_reason if response.prompt_feedback else 'N/A'}")
                 print(f"   Response Safety Ratings: {response.prompt_feedback.safety_ratings if response.prompt_feedback else 'N/A'}")
        except Exception as e:
             print(f"⚠️ Warning: Error processing quiz response: {e}")
             print(f"   Raw Quiz Response: {response}")

        return quiz_text

    except Exception as e:
        print(f"❌ An error occurred during Quiz generation API interaction: {e}")
        traceback.print_exc()
        return None


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Script ---")

    # --- Pre-run Checks ---
    if API_KEY == "YOUR_API_KEY" or not API_KEY: # Check if API key is set
        print("❌ Configuration Error: Please replace 'YOUR_API_KEY' with your actual Google API Key.")
        exit()

    if PDF_FILE_PATH == "PASTE_FULL_PATH_TO_YOUR_PDF_HERE" or not PDF_FILE_PATH:
        print("❌ Configuration Error: Please update the 'PDF_FILE_PATH' variable with the actual path to your PDF file.")
        exit()

    # --- Step 1: Extract text from PDF ---
    extracted_text = extract_text_from_pdf(PDF_FILE_PATH)

    if not extracted_text:
        print("\n--- Processing stopped because text could not be extracted from the PDF. ---")
        exit()

    print(f"\n✅ Text extraction complete. Total characters extracted: {len(extracted_text)}")
    # Optional: Print a small preview
    # print(f" Extracted text preview (first 500 chars):\n{extracted_text[:500]}...\n")

    # --- Step 2: Generate Course Outline from Extracted Text ---
    print("\n--- Proceeding to Generate Course Outline ---")
    generated_course_outline = generate_course_outline(API_KEY, MODEL_NAME, extracted_text)

    if not generated_course_outline:
        print("\n--- Course Outline generation failed or produced no output. Cannot proceed to quiz generation. Check logs above. ---")
        exit()

    print("\n" + "="*30 + " Generated Course Outline " + "="*30 + "\n")
    print(generated_course_outline)
    # print(to_markdown(generated_course_outline)) # Optional Markdown formatting

    # --- Step 3: Generate Quizzes from the Generated Course Outline ---
    print("\n--- Proceeding to Generate Quizzes from Course Outline ---")
    generated_quizzes = generate_quizzes_from_course(API_KEY, MODEL_NAME, generated_course_outline)

    if not generated_quizzes:
        print("\n--- Quiz generation failed or produced no output. Check logs above. ---")
        exit()

    print("\n" + "="*30 + " Generated Quizzes (20 Questions) " + "="*30 + "\n")
    print(generated_quizzes)
    # print(to_markdown(generated_quizzes)) # Optional Markdown formatting


    print("\n--- Script Finished ---")