import validators
import requests


def validate_and_check_url(url):
    # Validate URL format
    if not validators.url(url):
        return "Error: Invalid URL format."
    
    try:
        # Make a request to the URL
        response = requests.get(url)
        # Check if the status code is 200
        if response.status_code == 200:
            return "URL is valid and accessible, with status code 200."
        else:
            return f"Error: URL is accessible but returned status code {response.status_code}."
    except requests.exceptions.RequestException as e:
        # Handle exceptions for requests (e.g., connection errors)
        return f"Error: An error occurred while trying to access the URL, try with the correct URL. {e}"