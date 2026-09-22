import sys
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import GPT_CONFIG_124M 
from languagemodel import LanguageModel


model = LanguageModel(gpt_config=GPT_CONFIG_124M)

model.load_the_model()

model.generate_and_print_sample(start_context="Every effort moves you")
