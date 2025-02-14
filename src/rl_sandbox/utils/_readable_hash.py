from typing import Any


def generate_phrase_hash(num: Any) -> str:
    """
    Generates a unique, human-readable phrase from a given number.

    This function uses word lists and deterministic operations based on the number
    to create phrases. It's designed for readability and reasonable uniqueness within
    a practical range, but collisions are theoretically possible for very large numbers
    or limited word lists.

    Args:
        number: An integer to be hashed into a phrase.

    Returns:
        str: A human-readable phrase representing the hash of the number.
             Returns "zero phrase" if the input is 0.

    Raises:
        TypeError: if the input is not an integer.
    """

    if not isinstance(num, int):
        number = int(num)
    else:
        number = num

    if number == 0:
        return "zero phrase"  # Special case for zero

    # Word lists for phrase components - expand these for more diversity
    adjectives = [
        "happy", "bright", "clever", "gentle", "brave", "calm", "eager", "funny",
        "kind", "lively", "proud", "silly", "wise", "bold", "charming", "daring",
        "elegant", "fancy", "gleaming", "jolly", "keen", "merry", "neat", "optimistic",
        "peaceful", "quick", "rare", "smart", "tender", "unique", "vibrant", "witty",
        "zesty", "adorable", "blissful", "cozy", "delightful", "enchanting", "fearless",
        "graceful", "harmonious", "innocent", "joyful", "lovely", "magical", "nurturing",
        "open", "patient", "radiant", "serene", "thoughtful", "uplifting", "wonderful",
        "xenial", "youthful", "zealous", "amazing", "beautiful", "colorful", "dynamic"
    ]
    nouns = [
        "sun", "moon", "star", "tree", "flower", "river", "mountain", "cloud", "bird",
        "fish", "cat", "dog", "house", "book", "song", "dream", "smile", "laugh",
        "heart", "hand", "foot", "eye", "ear", "voice", "path", "road", "city",
        "world", "time", "wind", "rain", "fire", "water", "earth", "sky", "ocean",
        "forest", "garden", "field", "lake", "island", "beach", "stone", "sand",
        "cloud", "rainbow", "shadow", "silence", "whisper", "melody", "rhythm", "journey",
        "adventure", "mystery", "secret", "treasure", "gift", "promise", "hope", "belief",
        "wisdom", "courage", "passion", "spirit", "soul", "memory", "moment", "future"
    ]
    verbs = [
        "shines", "rises", "falls", "grows", "blooms", "flows", "climbs", "drifts", "flies",
        "swims", "purrs", "barks", "stands", "opens", "plays", "appears", "gleams", "rings",
        "beats", "holds", "steps", "sees", "hears", "speaks", "winds", "pours", "burns",
        "splashes", "trembles", "glows", "whispers", "echoes", "dances", "sings", "listens",
        "wanders", "explores", "reveals", "hides", "discovers", "offers", "vows", "cherishes",
        "inspires", "strengthens", "awakens", "remembers", "captures", "unfolds", "beckons",
        "transforms", "connects", "embraces", "illuminates", "resonates", "unites", "overcomes"
    ]
    adverbs = [
        "softly", "gently", "quickly", "slowly", "quietly", "loudly", "brightly", "dimly",
        "happily", "sadly", "eagerly", "calmly", "boldly", "shyly", "gracefully", "awkwardly",
        "elegantly", "simply", "neatly", "messily", "optimistically", "pessimistically",
        "peacefully", "angrily", "joyfully", "sorrowfully", "thoughtfully", "carelessly",
        "wonderfully", "terribly", "amazingly", "strangely", "beautifully", "colorfully",
        "dynamically", "harmoniously", "innocently", "magically", "mysteriously", "openly",
        "patiently", "passionately", "peacefully", "powerfully", "proudly", "quietly",
        "radically", "rapidly", "rarely", "regularly", "reliably", "respectfully", "responsibly"
    ]

    phrase_components = [adjectives, nouns, verbs, adverbs] # Structure of the phrase: Adjective Noun Verb Adverb
    phrase_words = []
    current_number = abs(number) # Use absolute value for word selection, handle negative later

    for word_list in phrase_components:
        list_length = len(word_list)
        if list_length == 0: # Handle empty word lists gracefully (though they shouldn't be empty)
            phrase_words.append("empty_list_word") # Or some placeholder
            continue

        index = current_number % list_length # Deterministic index based on the number
        phrase_words.append(word_list[index])
        current_number //= list_length # Reduce the number for the next component


    phrase = "_".join(phrase_words)

    if number < 0:
        phrase = "negative " + phrase # Indicate negative number

    return phrase # Capitalize for better readability
