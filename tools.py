
import os
import requests
from dotenv import load_dotenv
from agents import Agent, function_tool, OpenAIChatCompletionsModel
import json
from openai import AsyncOpenAI

# Import the tmdbsimple library
import tmdbsimple as tmdb

# Load keys from .env
load_dotenv()

# --- Initialize tmdbsimple with your API Key ---
tmdb.API_KEY = os.getenv("TMDB_API_KEY")

# Optionally, set a timeout (as per tmdbsimple docs)
tmdb.REQUESTS_TIMEOUT = 5 # seconds

# --- Configure OpenRouter Client (for poetry_agent's internal model) ---
openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# --- Tool 1: External API Call (TMDB - Movie Details) ---
@function_tool
def get_movie_details(movie_title: str, detail_requested: str = "all") -> str:
    """
    Retrieves and summarizes detailed information about a movie using TMDB API.
    This tool searches The Movie Database (TMDB) for the given movie title and returns
    key details such as its title, release date, genres, and a brief overview.

    Args:
        movie_title (str): The exact title of the movie to search for.
        detail_requested (str, optional): The specific movie detail the user is asking for.
                                          Valid options are "release_date", "genres", "overview", or "all" (default).

    Returns:
        str: A concise, human-readable summary of the movie's details.
             Returns an error message if the API key is missing,
             the movie is not found, or there's an API error.
    """
    if not tmdb.API_KEY:
        return "Error: TMDB API key is not configured. Please add TMDB_API_KEY to your .env file."

    try:
        search = tmdb.Search()
        response = search.movie(query=movie_title)

        if not search.results: # tmdbsimple stores results in search.results
            return f"No movie found with title: '{movie_title}'."

        # Take the first result (most relevant)
        first_movie_id = search.results[0]['id']

        movie = tmdb.Movies(first_movie_id)
        # Use movie.info() to populate attributes like title, overview, genres
        movie.info()

        title = movie.title if hasattr(movie, 'title') else "N/A"
        release_date = movie.release_date if hasattr(movie, 'release_date') else "N/A"
        overview = movie.overview if hasattr(movie, 'overview') else "No overview available."
        
        genres_list = [g['name'] for g in movie.genres if 'name' in g] if hasattr(movie, 'genres') and movie.genres else []
        genres = ", ".join(genres_list) if genres_list else "N/A"

        if detail_requested == "release_date":
            return f"The release date for '{title}' is: {release_date}."
        elif detail_requested == "genres":
            return f"The genres for '{title}' are: {genres}."
        elif detail_requested == "overview":
             return f"Overview of '{title}': {overview}"
        else: # detail_requested == "all" (or any other value)
            summary = (
                f"Here are the details for '{title}':\n"
                f"- Released: {release_date}\n"
                f"- Genres: {genres}\n"
                f"- Overview: {overview}"
            )
            return summary
    except tmdb.core.exc.TMDbException as e:
        # Catch specific tmdbsimple exceptions
        return f"An error occurred with the TMDB API: {e}"
    except requests.exceptions.RequestException as e:
        # Catch network-related errors
        return f"A network error occurred while contacting TMDB: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# --- Tool 2: Custom Agent-Based Tool (Poetry Agent) ---
poetry_agent = Agent(
    name="poetry_agent",
    instructions="You are a world-class poet. Your sole purpose is to write a short, beautiful poem based on the user's prompt. Do not do anything else.",
    model=OpenAIChatCompletionsModel(
        model="deepseek/deepseek-chat-v3-0324:free",
        openai_client=openrouter_client,
    ),
)