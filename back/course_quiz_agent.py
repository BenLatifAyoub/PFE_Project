import os
import time
import traceback
from typing import Optional, Dict, TypedDict

from langchain_google_genai import GoogleGenerativeAI
from langgraph.graph import Graph

# Try to import the API key from config.py, otherwise use environment variable or default
try:
    from config import GEMINI_API_KEY, GEMINI_MODEL_NAME
except ImportError:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")

# Define the state structure for LangGraph
class CourseQuizState(TypedDict):
    article_text: str
    course_outline: Optional[str]
    quiz: Optional[str]

class CourseQuizGenerationAgent:
    """
    Agent to generate a course outline and quiz questions based on article text
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
            print("❌ CourseQuiz Agent Error: GEMINI_API_KEY is not set in config.py or environment variables.")
            return

        try:
            print(f" CourseQuiz Agent: Initializing Gemini Model: {self.model_name}")
            self.model = GoogleGenerativeAI(model=self.model_name, google_api_key=self.api_key)
            self.is_available = True
            print("✅ CourseQuiz Agent: Gemini initialized successfully.")
        except Exception as e:
            print(f"❌ CourseQuiz Agent FATAL: Error initializing Google Generative AI: {e}")
            traceback.print_exc()
            self.model = None

    def _call_gemini(self, prompt: str, temperature: float = 0.5) -> Optional[str]:
        """
        Internal helper method to call the Gemini API via LangChain and handle responses/errors.
        """
        if not self.is_available or not self.model:
            print("❌ CourseQuiz Agent Error: Agent not available or model not initialized.")
            return None
        try:
            start_time = time.time()
            print(f" CourseQuiz Agent: Sending request to Gemini model '{self.model_name}'...")
            response = self.model.invoke(prompt)
            end_time = time.time()
            print(f" CourseQuiz Agent: Gemini response received in {end_time - start_time:.2f} seconds.")
            if not response:
                print(" CourseQuiz Agent WARNING: Gemini returned an empty response.")
                return "Error: Gemini returned an empty response."
            return response.strip()
        except Exception as e:
            print(f"❌ CourseQuiz Agent ERROR during Gemini API call: {e}")
            traceback.print_exc()
            return f"Error: Exception during Gemini API call - {type(e).__name__}"

    def generate_course_outline(self, article_text: str) -> Optional[str]:
        """
        Generates a structured course outline based on the provided article text.
        """
        if not article_text:
            print("❌ CourseQuiz Agent Error: No article text provided for course generation.")
            return None
        
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
        return self._call_gemini(course_prompt, temperature=0.4)

    def generate_quiz(self, course_outline_text: str) -> Optional[str]:
        """
        Generates 20 quiz questions based only on the provided course outline text.
        """
        if not course_outline_text or "Error:" in course_outline_text:
            print("❌ CourseQuiz Agent Error: No valid course outline text provided for quiz generation.")
            return None
        
        quiz_prompt = f"""
        Act as an assessment specialist. You are provided with a detailed course outline derived from a scientific article.
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
        return self._call_gemini(quiz_prompt, temperature=0.5)

    def generate_course_node(self, state: CourseQuizState) -> CourseQuizState:
        """
        LangGraph node to generate the course outline and update the state.
        """
        course_outline = self.generate_course_outline(state["article_text"])
        state["course_outline"] = course_outline
        return state

    def generate_quiz_node(self, state: CourseQuizState) -> CourseQuizState:
        """
        LangGraph node to generate the quiz based on the course outline and update the state.
        """
        if state["course_outline"] and "Error:" not in state["course_outline"]:
            quiz = self.generate_quiz(state["course_outline"])
            state["quiz"] = quiz
        else:
            state["quiz"] = "Error: Quiz generation skipped due to course failure."
        return state

    def build_graph(self):
        """
        Builds and compiles the LangGraph workflow for course and quiz generation.
        """
        graph = Graph()
        graph.add_node("generate_course", self.generate_course_node)
        graph.add_node("generate_quiz", self.generate_quiz_node)
        graph.add_edge("generate_course", "generate_quiz")
        graph.set_entry_point("generate_course")
        graph.set_finish_point("generate_quiz")
        return graph.compile()

    def generate_course_and_quiz(self, article_text: str) -> Optional[Dict[str, str]]:
        """
        Generates both the course outline and the quiz using a LangGraph workflow.

        :param article_text: The full text content of the article.
        :return: A dictionary {"course": outline_text, "quiz": quiz_text} or None if generation fails.
                 Returns error messages within the dict values if specific steps fail.
        """
        print(f" CourseQuiz Agent: Starting generation for article text (length: {len(article_text)} chars)...")
        start_gen_time = time.time()
        runnable = self.build_graph()
        initial_state = {"article_text": article_text, "course_outline": None, "quiz": None}
        final_state = runnable.invoke(initial_state)
        end_gen_time = time.time()
        print(f"✅ CourseQuiz Agent: Course and Quiz generation complete in {end_gen_time - start_gen_time:.2f} seconds.")
        return {"course": final_state["course_outline"], "quiz": final_state["quiz"]}