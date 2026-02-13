ALL_LANGUAGES = ['af', 'am', 'ar', 'az', 'bg', 'bn', 'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fr', 'he',
                 'hi', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'jv', 'ka', 'km', 'kn', 'ko', 'lt', 'ml', 'mr', 'nl',
                 'pl', 'pt', 'ru', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'ur', 'vi', 'zh']

TOPICS = ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"]

PROMPT = """Classify the following text into one of these topics: "science/technology", "travel", "politics", "sports", "health", "entertainment", "geography". Provide only the topic in English as your response.

text: `{}`
topic: `"""