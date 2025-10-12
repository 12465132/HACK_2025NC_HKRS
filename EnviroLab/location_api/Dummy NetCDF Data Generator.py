import os
import argparse
import json
import requests
import google.generativeai as genai

def find_nearest_place(lat, lon, api_key):
    """
    Finds the nearest infrastructure using Google Maps Places API.

    Args:
        lat (float): The latitude of the location.
        lon (float): The longitude of the location.
        api_key (str): The Google Maps API key.

    Returns:
        dict: A dictionary containing details of the nearest place, or None if not found.
    """
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    
    # We will search for a few common critical infrastructure types.
    # The API will rank results by distance from the provided location.
    # We are using a broad search keyword to capture various types of infrastructure.
    params = {
        'location': f"{lat},{lon}",
        'rankby': 'distance',
        'keyword': 'school,park,hospital,police,fire station,power station,water treatment',
        'key': api_key
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        results = response.json()

        if results.get("status") == "OK" and results.get("results"):
            # The first result is the nearest due to 'rankby=distance'
            return results["results"][0]
        else:
            print(f"Error from Places API: {results.get('status')} - {results.get('error_message', 'No results found.')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the Places API request: {e}")
        return None

def get_current_humidity(lat, lon):
    """
    Gets the current humidity for a given location using the Open-Meteo API.

    Args:
        lat (float): The latitude of the location.
        lon (float): The longitude of the location.

    Returns:
        int: The current relative humidity percentage, or None if an error occurs.
    """
    api_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=relative_humidity_2m"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        return data['current']['relative_humidity_2m']
    except requests.exceptions.RequestException as e:
        print(f"An error occurred fetching humidity data: {e}")
        return None
    except KeyError:
        print("Could not parse humidity from weather API response.")
        return None


def generate_infrastructure_analysis(place_name, place_type, humidity, api_key):
    """
    Generates a summary and analysis using the Gemini API.

    Args:
        place_name (str): The name of the infrastructure.
        place_type (str): The primary type of the infrastructure.
        humidity (int): The current relative humidity.
        api_key (str): The Gemini API key.

    Returns:
        dict: A dictionary containing the AI-generated analysis, or None if an error occurs.
    """
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return None
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        return None

    prompt = f"""
    Analyze the following piece of infrastructure based on the provided data.

    - Infrastructure Name: "{place_name}"
    - Infrastructure Type: "{place_type}"
    - Current Local Humidity: {humidity}%

    Based on this information, generate a valid JSON object with the following structure and keys:
    {{
        "ai_summary": "A concise, single-paragraph summary. Explain what this infrastructure is used for and how the specified high or low humidity might impact its operations, materials, or the people who use it.",
        "resources_used": ["A list of 2-3 primary resources this type of infrastructure typically consumes (e.g., electricity, water, fuel)."],
        "impact_reduction": ["A list of 2-3 actionable suggestions for how this type of infrastructure could reduce its environmental impact."],
        "is_critical": "A boolean value (true or false) indicating if this infrastructure type is generally considered critical for a community's function and safety."
    }}

    Provide only the raw JSON object in your response, with no additional text or markdown formatting.
    """
    
    try:
        response = model.generate_content(prompt)
        # Clean the response to ensure it's valid JSON
        cleaned_json_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_json_text)
    except Exception as e:
        print(f"An error occurred during Gemini API call: {e}")
        print(f"Received text: {response.text}")
        return None

def main():
    """
    Main function to run the infrastructure analysis script.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Find and analyze the nearest infrastructure for a given location.")
    parser.add_argument("latitude", type=float, help="The latitude of the location (e.g., 35.9132).")
    parser.add_argument("longitude", type=float, help="The longitude of the location (e.g., -79.0558).")
    args = parser.parse_args()

    # --- API Key Validation ---
    google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not google_maps_api_key:
        print("Error: GOOGLE_MAPS_API_KEY environment variable not set.")
        return
    if not gemini_api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    # --- Core Logic ---
    print(f"Searching for nearest infrastructure near ({args.latitude}, {args.longitude})...")
    
    nearest_place = find_nearest_place(args.latitude, args.longitude, google_maps_api_key)
    if not nearest_place:
        print("Could not find any nearby infrastructure. Exiting.")
        return

    place_name = nearest_place.get('name', 'N/A')
    place_type = nearest_place.get('types', ['unknown'])[0].replace('_', ' ').title()
    print(f"Found nearest place: {place_name} (Type: {place_type})")
    
    print("Fetching current climate data...")
    humidity = get_current_humidity(args.latitude, args.longitude)
    if humidity is None:
        print("Could not retrieve humidity data. Proceeding without it.")
        # We can still proceed, the AI will have less context.
        humidity = "not available"

    print("Generating AI analysis...")
    ai_analysis = generate_infrastructure_analysis(place_name, place_type, humidity, gemini_api_key)

    if not ai_analysis:
        print("Failed to generate AI analysis. Exiting.")
        return
        
    # --- Final JSON Output ---
    final_response = {
        "name": place_name,
        "type": place_type,
        **ai_analysis # Unpack the dictionary from Gemini
    }

    print("\n--- Analysis Complete ---")
    print(json.dumps(final_response, indent=4))


if __name__ == "__main__":
    main()
