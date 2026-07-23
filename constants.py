ALL_LANGUAGES = ['af', 'am', 'ar', 'az', 'bg', 'bn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fr', 'he',
                 'hi', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'jv', 'ka', 'km', 'kn', 'ko', 'lt', 'ml', 'mr', 'nl',
                 'pl', 'pt', 'ru', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'ur', 'vi', 'zh']

MUTUALLY_INTELLIGIBLE = {
    "bg": ["mk"],          # Bulgarian <-> Macedonian (very significantly)
    "mk": ["bg"],

    "pt": ["gl"],          # Portuguese <-> Galician (very significantly)
    "gl": ["pt"],

    "oc": ["ca"],          # Occitan <-> Catalan (significantly)
    "ca": ["oc"],

    "be": ["uk"],          # Belarusian <-> Ukrainian (significantly)
    "uk": ["be"],

    "hi": ["ur"],          # Hindi <-> Urdu (Hindustani, mutually intelligible)
    "ur": ["hi"],

    "id": ["ms"],          # Indonesian <-> Standard Malay (generally mutually intelligible)
    "ms": ["id"],

    "cs": ["sk"],          # Czech <-> Slovak
    "sk": ["cs"],

    "bs": ["hr", "sr"],    # Bosnian, Croatian, Serbian (Serbo-Croatian)
    "hr": ["bs", "sr"],
    "sr": ["bs", "hr"],
}

SENT_SUMM_TEMPLATE = "This sentence: \"{sent}\" means in one word:"
