# translation_agent_langgraph.py
import time
import traceback
from typing import Optional, Dict, TypedDict

try:
    # Use the specific version known to work more reliably
    from googletrans import Translator, LANGUAGES
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    Translator = None
    LANGUAGES = {}
    GOOGLETRANS_AVAILABLE = False
    print("WARNING: 'googletrans' library not found or import failed. Translation agent will be unavailable. Install with 'pip install googletrans==4.0.0-rc1'")

from langgraph.graph import StateGraph, END

# --- Original TranslationAgent class (kept mostly as is) ---
class TranslationAgent:
    """
    Agent responsible for translating text using the googletrans library
    (unofficial Google Translate API wrapper).
    """
    def __init__(self):
        """
        Initializes the TranslationAgent.
        """
        self.translator = None
        self.is_available = False

        if not GOOGLETRANS_AVAILABLE or not Translator:
            print("Translation Agent Error: 'googletrans' library is not available.")
            return # Stop initialization

        try:
            self.translator = Translator()
            # Optional: Perform a simple test translation to ensure it works
            # self.translator.translate("test", dest="es") # Be careful with rate limits if always on
            self.is_available = True
            print("Translation Agent (googletrans) Initialized Successfully.")
        except Exception as e:
            print(f"Translation Agent FATAL: Error initializing googletrans Translator: {e}")
            traceback.print_exc()
            self.is_available = False

    def translate(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> Dict[str, str]:
        """
        Translates the given text to the target language using googletrans.
        """
        if not self.is_available or not self.translator:
            return {"error": "Translation service is not available.", "details": "Agent not initialized correctly."}

        if not text or not isinstance(text, str) or not text.strip():
            return {"error": "Invalid input.", "details": "Text to translate cannot be empty."}

        if not target_lang or not isinstance(target_lang, str):
            return {"error": "Invalid input.", "details": "Target language code is required."}

        target_lang_code = target_lang.lower()
        source_lang_code = source_lang.lower() if source_lang else 'auto'

        if target_lang_code not in LANGUAGES and target_lang_code != 'auto':
             print(f"Warning: Target language '{target_lang_code}' not in googletrans.LANGUAGES. Proceeding anyway.")

        print(f"Translation Agent (googletrans): Requesting translation to '{target_lang_code}' (Source: '{source_lang_code}')...")
        start_time = time.time()

        try:
            result = self.translator.translate(
                text,
                dest=target_lang_code,
                src=source_lang_code
            )
            end_time = time.time()
            print(f"Translation Agent (googletrans): Translation successful in {end_time - start_time:.2f} seconds.")

            translated_text = result.text
            detected_source_lang = result.src

            print(f"Translation Agent (googletrans): Detected source language: {detected_source_lang}")
            return {"translated_text": translated_text, "detected_source_language": detected_source_lang}

        except Exception as e:
            print(f"Translation Agent (googletrans) Error: Unexpected error during translation: {e}")
            error_type = type(e).__name__
            print(f"Error Type: {error_type}")
            traceback.print_exc()
            details = f"An error occurred during translation ({error_type}). This might be due to network issues, changes in the Google Translate service, or request limits."
            if "JSONDecodeError" in error_type:
                details += " This often indicates Google blocked the request or changed its response format."
            return {"error": "Translation failed.", "details": details}

# --- LangGraph specific components ---

# 1. Define the State for the graph
class TranslationGraphState(TypedDict):
    text_to_translate: str
    target_language: str
    source_language: Optional[str] # Optional input

    # Outputs
    translated_text: Optional[str]
    detected_source_language: Optional[str]
    error_message: Optional[str]
    error_details: Optional[str]

# 2. Instantiate the original agent (globally or passed via config if preferred for more complex setups)
# For simplicity, we'll create one instance here.
# This ensures the translator is initialized only once.
_translation_agent_instance = TranslationAgent()

# 3. Define the Node function
def perform_translation_node(state: TranslationGraphState) -> Dict[str, Optional[str]]:
    """
    LangGraph node that calls the TranslationAgent to perform translation.
    Updates the state with the translation result or error information.
    """
    print("\n---NODE: PERFORM_TRANSLATION---")
    if not _translation_agent_instance.is_available:
        print("Translation node: Underlying TranslationAgent is not available.")
        return {
            "translated_text": None,
            "detected_source_language": None,
            "error_message": "Translation service unavailable",
            "error_details": "The googletrans library or service could not be initialized.",
        }

    text = state["text_to_translate"]
    target_lang = state["target_language"]
    source_lang = state.get("source_language") # .get() handles optional nicely

    print(f"Node input: Text='{text}', TargetLang='{target_lang}', SourceLang='{source_lang}'")

    result = _translation_agent_instance.translate(
        text=text,
        target_lang=target_lang,
        source_lang=source_lang
    )

    if "error" in result:
        print(f"Node result: Error - {result['error']}")
        return {
            "translated_text": None,
            "detected_source_language": None,
            "error_message": result["error"],
            "error_details": result.get("details"),
        }
    else:
        print(f"Node result: Success - '{result['translated_text']}'")
        return {
            "translated_text": result["translated_text"],
            "detected_source_language": result["detected_source_language"],
            "error_message": None, # Clear any previous error
            "error_details": None,
        }

# 4. Build the graph
def create_translation_graph():
    """
    Creates and compiles the LangGraph for translation.
    """
    workflow = StateGraph(TranslationGraphState)

    # Add the single node for translation
    workflow.add_node("translate_text", perform_translation_node)

    # Set the entry point
    workflow.set_entry_point("translate_text")

    # All paths lead to END after this node
    workflow.add_edge("translate_text", END)

    # Compile the graph
    app = workflow.compile()
    return app

