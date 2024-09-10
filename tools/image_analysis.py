import base64
from mimetypes import guess_type

from openai import AzureOpenAI

class AzureImageAnalyzer:
    def __init__(self, client: AzureOpenAI, deployment_name: str):
        """
        Initialize the AzureImageAnalyzer class with Azure OpenAI client configuration.

        Parameters:
        - client: Azure OpenAI instance
        """
       
        # Initialize the Azure OpenAI client
        self.client = client
        self.deployment_name = deployment_name

    async def local_image_to_data_url(self, image_path):
        """
        Encodes a local image to a data URL.

        Parameters:
        - image_path: Path to the image file

        Returns:
        - Data URL string
        """
        # Guess the MIME type of the image based on the file extension
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'  # Default MIME type if none is found

        # Read and encode the image file
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"

    async def analyze_image(self, arguments, max_tokens=300):
        """
        Analyze the image using the Azure OpenAI service.

        Parameters:
        - image_path: Path to the image file
        - max_tokens: Maximum number of tokens to return (default is 2000)

        Returns:
        - The response from the Azure OpenAI API in JSON format
        """
        # Get the data URL for the image
        data_url = await self.local_image_to_data_url(arguments["image_path"])

        # Prepare the messages with image data
        messages = [
            { "role": "system", "content": "As a leading AI expert in image analysis, you excel at describing and providing relevant information about the image provided based on the prompt. You have access to the local file system where the image is stored." },
            { "role": "user", "content": [  
                { 
                    "type": "text", 
                    "text": "Describe the image provided in detail with focus on this prompt: " + arguments["prompt"] 
                },
                { 
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }
            ]}
        ]

        # Send the request to the Azure OpenAI API
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            max_tokens=max_tokens
        )

        result = response.choices[0].message.content

        # Return the JSON response
        return result
