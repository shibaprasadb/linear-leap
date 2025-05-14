import google.generativeai as genai
import PIL.Image  # Import the PIL library
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

API_KEY="xyz"
plot = "yn"

def generate_text_with_image(prompt, image, api_key, model_name="gemini-2.0-flash"):
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
        # The key fix is in this structure
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
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def image_to_bytes(image):
    """
    Converts a PIL Image to a base64 encoded byte string.

    Args:
      image: PIL Image object

    Returns:
      base64 encoded byte string.
    """
    import base64
    buffered = io.BytesIO()
    # Check the image mode.  If it is RGBA, convert it to RGB.
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")  # Save as JPEG
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


def create_scatterplot():
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
    plt.figure(figsize=(8, 6))
    plt.scatter(square_footage, prices, alpha=0.7)
    plt.title("House Prices vs. Square Footage")
    plt.xlabel("Square Footage (sq ft)")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    return plt

def plot_to_pil(plot):
    """
    Converts a matplotlib.pyplot plot to a PIL Image object.

    Args:
        plot: A matplotlib.pyplot figure object.

    Returns:
        A PIL Image object of the plot.
    """
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plot.savefig(buf, format='png')
    buf.seek(0)  # Important: Seek back to the beginning of the buffer!
    plt.close(plot.gcf())  # Close the figure to free memory
    # Open the image from the BytesIO object using PIL
    img = PIL.Image.open(buf)
    return img


image = plot_to_pil(plot)

# Define the prompt
prompt = "Summarise these distributions.Check for these things: Are they multimodal, bimodal or normal? What could be the business implications of these? Summarise them in short bullet points. For each of these parameters make two bullet points: Shape and Business Implications"

# Generate text using the Gemini API
generated_text = generate_text_with_image(prompt, image, API_KEY, model_name="gemini-2.0-flash")