import google.generativeai as genai
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# Default API key - replace with your actual key or environment variable
API_KEY = "your_api_key_here"

def generate_text_with_image(prompt, image, api_key=API_KEY, model_name="gemini-2.0-flash"):
    """
    Generates text using the Gemini API, including an image.

    Args:
        prompt: The text prompt to send to the API.
        image: A PIL Image object.
        api_key: Your Gemini API key.
        model_name: The name of the Gemini model to use.
                    Defaults to "gemini-2.0-flash".
                    Must be a model that supports images.

    Returns:
        The generated text, or None on error.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        # Prepare the content with the image and text
        image_part = {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": image_to_bytes(image)
            }
        }
        
        text_part = {"text": prompt}
        
        response = model.generate_content([image_part, text_part])
        
        if response.text:
            return response.text
        else:
            print(f"Error: {response}")
            return "Error: No text was generated from the API. Please check your API key and quota."
    except Exception as e:
        error_message = str(e)
        # Provide more user-friendly messages for common errors
        if "invalid api key" in error_message.lower():
            return "Error: Invalid API key. Please provide a valid Gemini API key."
        elif "quota exceeded" in error_message.lower() or "rate limit" in error_message.lower():
            return "Error: API quota exceeded. Please try again later or use your own API key."
        elif "not available" in error_message.lower() and "model" in error_message.lower():
            return f"Error: The model '{model_name}' is not available. Please try a different model."
        else:
            return f"Error generating analysis: {error_message}"


def image_to_bytes(image):
    """
    Converts a PIL Image to a base64 encoded byte string.

    Args:
      image: PIL Image object

    Returns:
      base64 encoded byte string.
    """
    buffered = io.BytesIO()
    # Check the image mode.  If it is RGBA, convert it to RGB.
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")  # Save as JPEG
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


def create_sample_scatterplot():
    """
    Creates a sample scatter plot of house square footage vs. prices.

    Returns:
        A matplotlib.pyplot figure object of the plot.
    """
    # Generate sample data
    np.random.seed(42)  # for reproducibility
    square_footage = np.random.randint(1000, 3000, 50)
    prices = 100000 + square_footage * 150 + np.random.randint(-50000, 50000, 50)

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(square_footage, prices, alpha=0.7)
    ax.set_title("House Prices vs. Square Footage")
    ax.set_xlabel("Square Footage (sq ft)")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    return fig

def plot_to_pil(fig):
    """
    Converts a matplotlib.pyplot figure object to a PIL Image object.

    Args:
        fig: A matplotlib.pyplot figure object.

    Returns:
        A PIL Image object of the plot.
    """
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)  # Important: Seek back to the beginning of the buffer!
    plt.close(fig)  # Close the figure to free memory
    # Open the image from the BytesIO object using PIL
    img = PIL.Image.open(buf)
    return img

# Define a default prompt for analyzing plots
DEFAULT_PROMPT = """
Summarize these distributions.

Check for these things: Are they multimodal, bimodal or normal? What could be the business implications of these?

Summarize them in short bullet points.

For each of these parameters make two bullet points:
Shape
Business Implications
"""

# This code only runs when script is directly executed, not when imported
if __name__ == "__main__":
    # Example usage
    fig = create_sample_scatterplot()
    image = plot_to_pil(fig)
    generated_text = generate_text_with_image(DEFAULT_PROMPT, image, API_KEY)
    print(generated_text)