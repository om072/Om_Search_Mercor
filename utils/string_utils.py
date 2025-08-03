# utils/string_utils.py

# --- REMOVE: The incorrect 'import strings' line ---
# import strings # <-- DELETE THIS LINE

# import re # Only import 're' if you need regular expressions, otherwise delete this too

def parseCommaSeparatedIDs(idString: str) -> list[str]:
    """
    Converts a comma-separated string of IDs into a list of cleaned strings.

    This handles potential issues like leading/trailing spaces and multiple commas.

    Args:
        idString: A comma-separated string (e.g., "id1, id2,id3").

    Returns:
        A list of cleaned ID strings (e.g., ["id1", "id2", "id3"]).
    """
    # Use the .strip() method directly on the string object for trimming
    idString = idString.strip()

    # Check if the string is empty AFTER stripping
    if not idString:
        return []

    # Use the .split() method directly on the string object for splitting
    parts = idString.split(",")

    result = []
    for part in parts:
        # Use .strip() method on each part for trimming
        trimmed_part = part.strip()
        # Check if the trimmed part is not empty (handles "a,,b" cases)
        if trimmed_part:
            result.append(trimmed_part)

    return result

# --- You would add more general string utility functions here if needed ---
