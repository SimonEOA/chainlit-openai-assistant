

import random


def get_weather(location: str) -> str:
    """
    Dummy function to get a random number as weather data for the given location.
    """

    return f"Current weather in {location}: {random.randint(10, 28)}°C"
    
    