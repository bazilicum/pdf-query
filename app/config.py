
# Setting the right sentence trasformer will determine the quality of the vector space and the ability to process multilingual prompts. Some of the options you can use:
# SENTENCE_TRANSFORMER = 'all-MiniLM-L6-v2'  # English only 
# SENTENCE_TRANSFORMER = 'intfloat/multilingual-e5-large'  # Multilengual heavy
# SENTENCE_TRANSFORMER = 'distiluse-base-multilingual-cased-v2' # Multilengual medium
SENTENCE_TRANSFORMER = 'all-MiniLM-L6-v2'  # English only 
OPEN_AI_MODEL = 'gpt-4o-mini' #go to https://platform.openai.com/docs/models for the full supported model list and the cost
DB_ADDR = "http://qdrant"
DB_PORT = 6333
WEB_SERVER_HOST = '0.0.0.0'
WEB_SERVER_PORT = 5001
