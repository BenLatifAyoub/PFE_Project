import os
import time
import traceback
from typing import Optional, Dict, TypedDict, List

from langchain_google_genai import GoogleGenerativeAI
from langgraph.graph import Graph

# Try to import the API key from config.py, otherwise use environment variable or default
try:
    from config import GEMINI_API_KEY, GEMINI_MODEL_NAME
except ImportError:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyACQwN6IEzGeB59hUvVhGwpWJQiaHr5q9k") # Replace with your actual key if not using config/env
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")

# Define the state structure for LangGraph (can be extended if needed)
class CourseQuizState(TypedDict):
    article_text: Optional[str] # For single article flow
    article_texts: Optional[List[str]] # For multi-article flow
    course_themes: Optional[str] # Identified themes from multiple articles
    selected_points: Optional[str] # User selected points for course generation
    course_outline: Optional[str]
    quiz: Optional[str]

class AdvancedCourseGeneratorAgent:
    """
    Agent to:
    1. Identify common themes from multiple articles.
    2. Generate a course outline based on selected themes and multiple articles.
    3. Generate a course outline and quiz from a single article (original functionality).
    using LangChain with Google Gemini API and LangGraph for workflow management.
    """
    def __init__(self, api_key: Optional[str] = GEMINI_API_KEY, model_name: str = GEMINI_MODEL_NAME):
        """
        Initializes the agent with the Gemini API key and model name.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.model = None
        self.is_available = False

        if not self.api_key:
            print("❌ AdvancedCourseGeneratorAgent Error: GEMINI_API_KEY is not set.")
            # You might want to raise an exception here if the API key is critical
            # For demonstration, we'll allow it to proceed but flag as not available.
            # raise ValueError("GEMINI_API_KEY is not set.")
            return

        try:
            print(f" AdvancedCourseGeneratorAgent: Initializing Gemini Model: {self.model_name}")
            self.model = GoogleGenerativeAI(model=self.model_name, google_api_key=self.api_key)
            self.is_available = True
            print("✅ AdvancedCourseGeneratorAgent: Gemini initialized successfully.")
        except Exception as e:
            print(f"❌ AdvancedCourseGeneratorAgent FATAL: Error initializing Google Generative AI: {e}")
            traceback.print_exc()
            self.model = None

    def _call_gemini(self, prompt: str) -> Optional[str]:
        """
        Internal helper method to call the Gemini API via LangChain and handle responses/errors.
        """
        if not self.is_available or not self.model:
            print("❌ AdvancedCourseGeneratorAgent Error: Agent not available or model not initialized.")
            return "Error: Agent not available or model not initialized."
        try:
            start_time = time.time()
            print(f" AdvancedCourseGeneratorAgent: Sending request to Gemini model '{self.model_name}'...")
            # Ensure the prompt is a string, Langchain expects a string for single prompt invocation
            if not isinstance(prompt, str):
                print(f" AdvancedCourseGeneratorAgent WARNING: Prompt is not a string, type: {type(prompt)}. Converting to string.")
                prompt = str(prompt)

            response = self.model.invoke(prompt) # Pass temperature here
            end_time = time.time()
            print(f" AdvancedCourseGeneratorAgent: Gemini response received in {end_time - start_time:.2f} seconds.")
            
            # Langchain GoogleGenerativeAI returns a string directly (content of the AIMessage)
            if not response:
                print(" AdvancedCourseGeneratorAgent WARNING: Gemini returned an empty response.")
                return "Error: Gemini returned an empty response."
            return response.strip()
        except Exception as e:
            print(f"❌ AdvancedCourseGeneratorAgent ERROR during Gemini API call: {e}")
            traceback.print_exc()
            return f"Error: Exception during Gemini API call - {type(e).__name__}: {str(e)}"

    def _combine_articles_for_prompt(self, article_texts: List[str]) -> str:
        """Helper to format multiple articles for a prompt."""
        if not article_texts:
            return ""
        combined = ""
        for i, text in enumerate(article_texts):
            combined += f"--- ARTICLE {i+1} START ---\n{text}\n--- ARTICLE {i+1} END ---\n\n"
        return combined

    # --- NEW FUNCTION 1: Identify Similar Points/Themes ---
    def identify_course_themes_from_articles(self, article_texts: List[str]) -> Optional[str]:
        """
        Identifies similar points or themes across a list of article texts
        that could be used to structure a unified course.
        """
        if not self.is_available:
            return "Error: Agent not available."
        if not article_texts:
            print("❌ AdvancedCourseGeneratorAgent Error: No article texts provided for theme identification.")
            return "Error: No article texts provided."

        combined_articles = self._combine_articles_for_prompt(article_texts)
        if not combined_articles:
             return "Error: Combined articles text is empty."

        prompt = f"""
        You are an expert curriculum developer tasked with analyzing multiple research articles to find common ground for a new, unified course.
        Carefully read the following {len(article_texts)} article texts provided below.

        {combined_articles}

        Based STRICTLY on the content of these articles, identify 5-7 key overlapping themes, concepts, or significant findings that are present across multiple articles.
        These themes should be distinct yet interconnected enough to form the main modules or sections of a cohesive educational course.

        For each identified theme, provide:
        1. A concise title for the theme (e.g., "Mechanisms of [X]", "Applications of [Y] in [Z]", "Ethical Considerations in [A]").
        2. A brief 1-2 sentence description explaining the theme and hinting at which articles contribute to it (without needing to cite specific article numbers, just generally).

        Present your findings as a numbered list of themes.
        Example:
        1. Theme Title: Core Principles of Quantum Entanglement
           Description: This theme explores the fundamental concepts of quantum entanglement, drawing on discussions of Bell's theorem and experimental verifications presented across the articles.
        2. Theme Title: Applications in Quantum Computing
           Description: Several articles touch upon how entanglement is a critical resource for quantum computation, particularly in qubit manipulation and quantum algorithm development.

        Return ONLY the list of themes and their descriptions.
        """
        print(" AdvancedCourseGeneratorAgent: Identifying course themes...")
        return self._call_gemini(prompt)

    # --- NEW FUNCTION 2: Create Course from Selected Points & Articles ---
    def generate_course_from_themes(self, article_texts: List[str], selected_themes_text: str) -> Optional[str]:
        """
        Generates a full course outline based on a list of article texts and pre-selected themes.
        The course structure (overview, outcomes, modules, etc.) will be similar to the
        original `generate_course_outline` method.
        """
        if not self.is_available:
            return "Error: Agent not available."
        if not article_texts:
            print("❌ AdvancedCourseGeneratorAgent Error: No article texts provided for course generation.")
            return "Error: No article texts provided."
        if not selected_themes_text:
            print("❌ AdvancedCourseGeneratorAgent Error: No selected themes provided for course generation.")
            return "Error: No selected themes provided."

        combined_articles = self._combine_articles_for_prompt(article_texts)
        if not combined_articles:
             return "Error: Combined articles text is empty."

        course_prompt = f"""
        You are an expert instructional designer and subject-matter expert.
        Your mission is to generate a fully developed online course (script only—no slide decks) based strictly on the **Selected Course Themes** and the **Source Article Texts** provided below. Do NOT add any content beyond what appears in these articles. Synthesize across the articles where appropriate.

        ---

        **Selected Course Themes:**
        {selected_themes_text}

        ---

        **Source Article Texts ({len(article_texts)} articles):**
        {combined_articles}

        ---

        Please respond in Markdown with the following structure:

        ## 1. Course Overview
        - **Title**: A concise, descriptive course name reflecting the themes and article content.
        - **Target Audience**: E.g. “Researchers interested in interdisciplinary approaches to X.”
        - **Prerequisites**: What learners should know or be able to do beforehand.
        - **Course Goal**: One sentence capturing the overall aim, tied directly to the Selected Course Themes.

        ## 2. Learning Outcomes
        - **5–7** action-oriented outcomes (e.g. “Synthesize X from multiple perspectives,” “Critically evaluate Y using evidence from the articles”) mapped to the themes and article content.

        ## 3. Modules
        Create **3–5** modules, each centred on a major theme (or a cluster of related themes):

        For each module:

        1. **Module Title**  
        – Drawn from or closely related to one Selected Course Theme.

        2. **Duration**  
        – Estimate time (e.g. “1.5–2 hours”).

        3. **Module Objectives**  
        – 2–3 specific objectives aligned with the overall Learning Outcomes.

        4. **Lecture Script**  
        – ~5–7 paragraphs of narrative. **Within the script, explicitly reference each source article by its title** whenever you draw on it: e.g. “According to *Article Title A*, …”. If multiple articles inform the same point, list each, e.g.: “*(Article Title A; Article Title B)*”.

        5. **Key Takeaways**  
        – 2–3 bullets summarising the module’s core messages.

        6. **Discussion Prompts** (optional)  
        – 1–2 open-ended questions linking back to the articles and themes.

        ## 4. Glossary
        - Define **8–10** key terms as they appear across the articles, especially those central to the themes.

        ## 5. Consolidated Bibliography (optional)
        - If the articles include references, list **3–5** most foundational or frequently cited works. If no bibliographies are present, state: “Key concepts are drawn directly from the provided article texts.”

        ## 6. Teaching Tips
        - 3–4 sentences on best practices for teaching this course, with emphasis on synthesising and comparing article perspectives.

        ---

        ### Design & Quality Checklist
        - **Alignment**: All components must map cleanly to the Selected Course Themes.  
        - **Synthesis**: Combine insights from multiple articles in each module’s script.  
        - **Source Adherence**: Do not introduce any outside materials.  
        - **Clarity & Cohesion**: Ensure logical progression from one module to the next.

        Generate the full course now.
        """

        print(f" AdvancedCourseGeneratorAgent: Generating course from {len(article_texts)} articles and selected themes...")
        return self._call_gemini(course_prompt)

    # --- Original Functionality (Single Article Course & Quiz) ---
        """
        Generates a structured course outline based on a single provided article text.
        (This is the original `generate_course_outline` method, renamed for clarity)
        """
        if not self.is_available:
            return "Error: Agent not available."
        if not article_text:
            print("❌ AdvancedCourseGeneratorAgent Error: No article text provided for single-article course generation.")
            return "Error: No article text provided for single-article course generation."
        
        # Using the original prompt structure for single article
        course_prompt = f"""
        You are an expert instructional designer AND subject‐matter expert. Read the scientific article text below STRICTLY as source material—do NOT add content beyond it. Your mission: generate a fully-developed online course (lecture script only—no slide decks) that follows a clear five‐step design and avoids common pitfalls.

        ---  
        **Full Article Text:**  
        {article_text}  
        ---

        Your response must use Markdown and include:

        1. ## Course Overview  
        - **Title**: a concise, informative course name.  
        - **Target Audience**: who this course serves (e.g. “Advanced neuroscience grad students”).  
        - **Prerequisites**: what students must already know or be able to do before enrolling.  
        - **Course Goal**: one sentence capturing the course’s overall aim.

        2. ## Learning Outcomes  
        - List **5–7** action-oriented outcomes (e.g. “Analyze X,” “Design Y”) that map directly to the article’s concepts, methods, and findings.

        3. ## Modules (4–6 sessions)  
        For each module:
        1. **Title**  
        2. **Duration**: 1.5–2 hours  
        3. **Module Objectives**: 2–3 outcomes drawn from your Learning Outcomes.  
        4. **Lecture Script**: a ~5–7-paragraph narrative explaining that section of the paper in depth.  
        5. **Quiz**: 5 questions (mix MCQ + short answer) with answers—administered online at the end of the module.  
        6. **Resources**: list relevant figures/tables from the article and 1–2 quick reference sheets (e.g. vocabulary handout).

        4. ## Glossary  
        - Define **8–10** key terms as used in the article.

        5. ## Bibliography  
        - Cite **3–5** references drawn **only** from the article’s own bibliography.

        6. ## Teaching Tips  
        - **3–4 sentences** on pacing, group-discussion alternatives for online, and pitfalls to avoid.

        ---

        ### Design & Quality Checklist

        - **Step 1 (Audience & Prerequisites)**: tailor objectives & activities to who they are and what they already know  
        - **Step 2 (Objectives)**: clear, measurable outcomes up front  
        - **Step 3 (Modules)**: bite-sized, logically sequenced topics  
        - **Step 4 (Activities)**: quizzes only—no overload  
        - **Step 5 (Assessments)**: spaced quizzes at end of each module

        **Avoid**: vague objectives, content overload, ignoring prerequisites, and sparse assessments.
        """
        print(" AdvancedCourseGeneratorAgent: Generating single-article course outline...")
        return self._call_gemini(course_prompt)

    def generate_quiz_from_outline(self, course_outline_text: str) -> Optional[str]:
        """
        Generates 20 quiz questions based only on the provided course outline text.
        (This is the original `generate_quiz` method, renamed for clarity)
        """
        if not self.is_available:
            return "Error: Agent not available."
        if not course_outline_text or "Error:" in course_outline_text: # Check for upstream errors
            print("❌ AdvancedCourseGeneratorAgent Error: No valid course outline text provided for quiz generation.")
            return "Error: No valid course outline text for quiz generation."
        
        quiz_prompt = f"""
        Act as an assessment specialist. You are provided with a detailed course outline.
        Your task is to create exactly 20 quiz questions based strictly and solely on the information presented in this course outline. Do not refer back to any original article or external knowledge.

        The quiz must adhere to the following requirements:
        Total Questions: Exactly 20 questions.
        Source Material: Base all questions and answers only on the provided course outline text below.
        Question Types: Include a mix of question types covering different aspects of the course outline. Aim for approximately:
        8 Multiple Choice (MC) questions (with 4 distinct options: A, B, C, D).
        7 True/False (TF) questions.
        5 Short Answer (SA) questions (requiring a brief, factual answer based on the outline).
        Content Focus: Questions should test understanding of the learning objectives, key concepts from modules, definitions of key terminology, and the overall summary points mentioned in the course outline.
        Clarity: Questions should be clear, unambiguous, and directly answerable from the provided text.
        Answers: Provide the correct answer immediately after each question.
        For MC: Clearly indicate the correct option (e.g., "Answer: C").
        For TF: Clearly state "Answer: True" or "Answer: False".
        For SA: Provide a concise, accurate model answer based only on the course outline content (e.g., "Answer: [Brief factual answer from outline]").
        Source Reference: After each answer, include the specific source reference from the course outline (e.g., section heading, direct phrase, or sentence number if available) that justifies the answer. Use this format:
        Source Reference: "[Exact quote or location from the course outline text]"
        Formatting:
        Number each question sequentially from 1 to 20.
        Clearly label the question type in parentheses after the number (e.g., "1. (MC)", "9. (TF)", "16. (SA)").
        Format options for MC questions clearly ( A), B), C), D) ).

        Course Outline Text:
        --- START OF COURSE OUTLINE TEXT ---
        {course_outline_text}
        --- END OF COURSE OUTLINE TEXT ---

        Generate the 20 quiz questions with answers and source references now.
        """
        print(" AdvancedCourseGeneratorAgent: Generating quiz from outline...")
        return self._call_gemini(quiz_prompt)


    def _generate_quiz_node(self, state: CourseQuizState) -> CourseQuizState:
        """ LangGraph node for quiz generation (works for any outline). """
        if "course_outline" not in state or not state["course_outline"] or "Error:" in state["course_outline"]:
            state["quiz"] = "Error: Quiz generation skipped due to missing or failed course outline."
            return state
        quiz = self.generate_quiz_from_outline(state["course_outline"])
        state["quiz"] = quiz
        return state


# --- Example Usage ---
if __name__ == "__main__":
    # Ensure GEMINI_API_KEY is set in your environment or config.py
    # For demonstration, you might need to replace None with your key if it's not picked up
    # from config import GEMINI_API_KEY (if you have this file and key in it)
    # api_key = GEMINI_API_KEY
    api_key = "AIzaSyACQwN6IEzGeB59hUvVhGwpWJQiaHr5q9k"
    if not api_key:
        print("GEMINI_API_KEY not found. Please set it as an environment variable or in config.py")
        # Example: api_key = "YOUR_ACTUAL_GEMINI_API_KEY" # Not recommended for production
        # For this example to run, you NEED a valid API key.
        # exit() # or handle this gracefully

    agent = AdvancedCourseGeneratorAgent(api_key=api_key)

    if not agent.is_available:
        print("Agent could not be initialized. Exiting example.")
        exit()

    # --- Dummy Article Texts for Demonstration ---
    article1_text = """
    Photosynthesis in plants is a critical process converting light energy into chemical energy.
    It primarily occurs in chloroplasts, utilizing chlorophyll to capture sunlight.
    The two main stages are the light-dependent reactions and the Calvin cycle (light-independent reactions).
    Light reactions produce ATP and NADPH, which then power the Calvin cycle to fix CO2 into sugars.
    Factors affecting photosynthesis include light intensity, CO2 concentration, and temperature.
    Understanding this process is vital for agriculture and addressing climate change.
    Key terms: Chlorophyll, ATP, NADPH, Calvin Cycle, CO2 fixation.
    References: Smith et al. (2020) Journal of Plant Physiology.
    """

    article2_text = """
    Cellular respiration is the metabolic process by which organisms obtain energy from organic molecules.
    It occurs in mitochondria in eukaryotes and involves glycolysis, the Krebs cycle, and oxidative phosphorylation.
    Glycolysis breaks down glucose into pyruvate. The Krebs cycle further oxidizes pyruvate derivatives.
    Oxidative phosphorylation, involving the electron transport chain, generates the majority of ATP.
    This process is complementary to photosynthesis. While photosynthesis stores energy, respiration releases it.
    Key terms: Mitochondria, Glycolysis, Krebs Cycle, Oxidative Phosphorylation, ATP.
    References: Jones et al. (2019) Journal of Cellular Biology.
    """

    article3_text = """
    Bioenergetics encompasses the study of energy flow through living systems.
    Photosynthesis and cellular respiration are cornerstone processes in bioenergetics.
    ATP (adenosine triphosphate) serves as the universal energy currency in cells, produced by one process and consumed by another.
    The efficiency of energy conversion is a key research area, with implications for biotechnology and metabolic engineering.
    Electron transport chains are fundamental to both photosynthesis (in chloroplasts) and respiration (in mitochondria).
    Key terms: Bioenergetics, Energy Conversion, ATP, Electron Transport Chain.
    References: Lee et al. (2021) Annual Review of Biochemistry.
    """

    list_of_articles = [article1_text, article2_text, article3_text]

    print("\n--- 1. Identifying Course Themes from Multiple Articles ---")
    identified_themes = agent.identify_course_themes_from_articles(list_of_articles)
    if identified_themes and "Error:" not in identified_themes:
        print("\nIdentified Course Themes:")
        print(identified_themes)

        print("\n--- 2. Generating Course from Identified Themes ---")
        # For this example, we'll use the themes directly as identified.
        # In a real application, you might allow the user to select or edit these themes.
        selected_points_for_course = identified_themes
        
        full_course_from_themes = agent.generate_course_from_themes(list_of_articles, selected_points_for_course)
        if full_course_from_themes and "Error:" not in full_course_from_themes:
            print("\nGenerated Course from Themes:")
            print(full_course_from_themes)
            
            # Optional: Generate quiz for this new course outline
            # print("\n--- Generating Quiz for Multi-Article Course Outline ---")
            # quiz_for_themed_course = agent.generate_quiz_from_outline(full_course_from_themes)
            # if quiz_for_themed_course:
            #     print("\nQuiz for Themed Course:")
            #     print(quiz_for_themed_course)

        else:
            print(f"\nFailed to generate course from themes or themes were not identified properly: {full_course_from_themes}")
    else:
        print(f"\nFailed to identify themes: {identified_themes}")
