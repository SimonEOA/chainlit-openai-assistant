import io
import json
import os
from io import BytesIO
from pathlib import Path
from typing import List
import plotly


from openai import AsyncAzureOpenAI, AzureOpenAI, AsyncAssistantEventHandler

from literalai.helper import utc_now

import chainlit as cl
from chainlit.config import config
from chainlit.element import Element

from tools.get_weather import get_weather
from openai.types.beta.threads.runs import RunStep

import pandas
import json

from tools.image_analysis import AzureImageAnalyzer



AZURE_OPENAI_API_VERSION = os.environ.get('AZURE_OPENAI_API_VERSION')
AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.environ.get('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME')

async_openai_client = AsyncAzureOpenAI(api_key=AZURE_OPENAI_API_KEY, api_version=AZURE_OPENAI_API_VERSION, azure_endpoint=AZURE_OPENAI_ENDPOINT)
sync_openai_client = client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,  
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint = AZURE_OPENAI_ENDPOINT
    )

ImageAnalyzer = AzureImageAnalyzer(client=sync_openai_client, deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME)

available_functions = {"get_weather": get_weather, "image_analysis": ImageAnalyzer.analyze_image}
verbose_output = True

assistant = client.beta.assistants.create(
    name="Axel AI",
    model="gpt-4o-2024-05-13",  # Replace with your model deployment name.
    instructions="You are a helpful product support assistant and you answer questions based on the files provided to you.",
    tools=[
        {"type": "file_search"},
        {"type": "code_interpreter"},
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Determine weather in my location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state e.g. Seattle, WA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["c", "f"]
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "image_analysis",
                "description": "Analyze an image",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to use for image analysis, constructed by the assistant based on context and user input"
                        }
                    },
                    "required": ["prompt"]
                }
            }
        }
    ],
    tool_resources={
        "file_search": {"vector_store_ids": ["vs_enHy9WWIxQXbgZM5WIcdTyCd"]},
        "code_interpreter": {"file_ids": []}
    },
    temperature=1,
    top_p=1
)



config.ui.name = assistant.name
class EventHandler(AsyncAssistantEventHandler):

    def __init__(self, assistant_name: str) -> None:
        super().__init__()
        self.current_message: cl.Message = None
        self.current_step: cl.Step = None
        self.current_tool_call = None
        self.assistant_name = assistant_name

    async def on_text_created(self, text) -> None:
        self.current_message = await cl.Message(author=self.assistant_name, content="").send()

    async def on_text_delta(self, delta, snapshot):
        await self.current_message.stream_token(delta.value)

    async def on_text_done(self, text):
        await self.current_message.update()
        if text.annotations:
            print("annotations", text.annotations)
            file = await async_openai_client.files.with_raw_response.content(text.annotations[0].file_citation.file_id)
            print(file)
            for annotation in text.annotations:
                if annotation.type == "file_path":
                    response = await async_openai_client.files.with_raw_response.content(annotation.file_path.file_id)
                    file_name = annotation.text.split("/")[-1]
                    try:
                        fig = plotly.io.from_json(response.content)
                        element = cl.Plotly(name=file_name, figure=fig)
                        await cl.Message(
                            content="",
                            elements=[element]).send()
                    except Exception as e:
                        element = cl.File(content=response.content, name=file_name)
                        await cl.Message(
                            content="",
                            elements=[element]).send()
                    # Hack to fix links
                    if annotation.text in self.current_message.content and element.chainlit_key:
                        self.current_message.content = self.current_message.content.replace(annotation.text, f"/project/file/{element.chainlit_key}?session_id={cl.context.session.id}")
                        await self.current_message.update()

    async def on_tool_call_created(self, tool_call):
        if tool_call.type == "code_interpreter":
            self.current_tool_call = tool_call.id
            self.current_step = cl.Step(name=tool_call.type, type="tool")
            self.current_step.language = "python"
            self.current_step.created_at = utc_now()
            await self.current_step.send()
        elif tool_call.type == "file_search":
            self.current_step = cl.Step(name=tool_call.type, type="tool")
            self.current_step.language = "markdown"
            self.current_step.created_at = utc_now()
            await self.current_step.send()
        elif tool_call.type == "function":
            self.current_step = cl.Step(name=tool_call.function.name, type="tool")
            self.current_step.created_at = utc_now()
            await self.current_step.send()

    async def on_tool_call_delta(self, delta, snapshot):   

        if delta.type == "code_interpreter":
            if delta.code_interpreter.outputs:
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        error_step = cl.Step(
                            name=delta.type,
                            type="tool"
                        )
                        error_step.is_error = True
                        error_step.output = output.logs
                        error_step.language = "markdown"
                        error_step.start = self.current_step.start
                        error_step.end = utc_now()
                        await error_step.send()
            else:
                if delta.code_interpreter.input:
                    await self.current_step.stream_token(delta.code_interpreter.input)
            

    async def on_event(self, event):
        thread_id = cl.user_session.get("thread_id")

        if event.event == 'thread.run.requires_action':

            if event.data.status == "requires_action":
                tool_responses = []
                if (
                    event.data.required_action.type == "submit_tool_outputs"
                    and event.data.required_action.submit_tool_outputs.tool_calls is not None
                ):
                    tool_calls = event.data.required_action.submit_tool_outputs.tool_calls


                    self.current_step = cl.Step(name="functions", type="tool") 
                    self.current_step.created_at = utc_now()       

                    for call in tool_calls:
                        if call.type == "function":
                            if call.function.name not in available_functions:
                                raise Exception("Function requested by the model does not exist")
                            
                            print("Function call:", call.function.name)
                            function_to_call = available_functions[call.function.name]
                            function_arguments = json.loads(call.function.arguments)
                            if call.function.name == "image_analysis":
                                # add image path from context to arguments
                                function_arguments["image_path"] = cl.user_session.get("image_path")
                            tool_response = await function_to_call(function_arguments)
                            tool_responses.append({"tool_call_id": call.id, "output": tool_response})
                    
                    self.current_step.output = tool_responses
                    await self.current_step.send()

                    async with async_openai_client.beta.threads.runs.submit_tool_outputs_stream(
                        thread_id=thread_id,
                        run_id=event.data.id,
                        tool_outputs=tool_responses,
                        event_handler=EventHandler(assistant_name=self.assistant_name),
                    ) as stream:
                        await stream.until_done()
                    


    async def on_tool_call_done(self, tool_call):
        if tool_call.type == "code_interpreter":
            self.current_step.end = utc_now()
            await self.current_step.update()
        elif tool_call.type == "file_search":
            self.current_step.end = utc_now()
            await self.current_step.update()


    async def on_image_file_done(self, image_file):
        image_id = image_file.file_id
        response = await async_openai_client.files.with_raw_response.content(image_id)
        image_element = cl.Image(
            name=image_id,
            content=response.content,
            display="inline",
            size="large"
        )
        if not self.current_message.elements:
            self.current_message.elements = []
        self.current_message.elements.append(image_element)
        await self.current_message.update()


@cl.step(type="tool")
async def speech_to_text(audio_file):
    response = await async_openai_client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )

    return response.text


async def upload_files(files: List[Element]):
    file_ids = []
    for file in files:
        purpose = "assistants"
        uploaded_file = await async_openai_client.files.create(
            file=Path(file.path), purpose=purpose
        )
        file_ids.append(uploaded_file.id)
    
    return file_ids

async def process_files(files: List[Element]):
    # Upload files and get file_ids
    file_ids = []
    if len(files) > 0:
        file_ids = await upload_files(files)

    # Prepare the content and attachments
    content = []
    tools = []

    for file_id, file in zip(file_ids, files):
        if file.mime.startswith("image/"):
            # save image path to context
            cl.user_session.set("image_path", file.path)
            
        
        # Add non-image files to tools for handling like before
        tools.append({
            "file_id": file_id,
            "tools": [{"type": "file_search"}] if file.mime in [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                "text/markdown", 
                "application/pdf", 
                "text/plain"
            ] else [{"type": "code_interpreter"}],
        })
    
    return tools, content

def init_thread_vector_store():

    try:
        vector_store = client.beta.vector_stores.retrieve("vs_9pHSQlWdoc9zjjRElnfsXPsc")

    except:
        vector_store = client.beta.vector_stores.create(
            name="user_thread_vector_store",
            expires_after= {
                "anchor": "last_active_at",
                "days": 1
                },
            )
        
    cl.user_session.set("thread_vector_store_id", vector_store.id)
    
    return vector_store

def add_files_to_vector_store(file_streams):

    init_thread_vector_store_id = cl.user_session.get("thread_vector_store_id")
    
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=init_thread_vector_store_id,
        files=file_streams,
        )
    
    return file_batch

  

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Get the current weather",
            message="Get the current weather in a specific location.",
            icon="/public/write.svg",
            ),
        cl.Starter(
            label="What is Axnine?",
            message="What is Axnine?",
            icon="/public/write.svg",
            )
        ]


@cl.on_chat_start
async def start_chat():
    
    
    thread_vector_store = init_thread_vector_store()
    
    
    # Create a Thread
    thread = await async_openai_client.beta.threads.create(
        #tool_resources={"file_search": {"vector_store_ids": [thread_vector_store.id]}},
    )

    
    # Store thread ID in user session for later use
    cl.user_session.set("thread_id", thread.id)
    


#@cl.on_chat_end
#async def end_chat():
    # Delete the assistant
    #await async_openai_client.beta.assistants.delete(assistant.id)
    

@cl.on_message
async def main(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")

    """
    file_streams = []
    image_files = []
    json_files = []  # List to store JSON file content

    files = message.elements

    for file in files:
        file_path = Path(file.path)
        
        # Check if the file is a document or text-based file
        if file.mime in [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",        # .xlsx
            "application/pdf",                                                          # .pdf
            "text/plain",                                                               # .txt
            "text/markdown",                                                            # .md
            "application/json"                                                          # .json
        ]:
            await cl.Message(
                author=assistant.name,
                content="Uploading file... with mime type: " + file.mime,
            ).send()

            # Check if the file is a binary or text-based file
            if file.mime in [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",        # .xlsx
                "application/pdf",                                                          # .pdf
                "application/json"                                                          # .json
            ]:
                
                file_streams.append(open(file_path, "rb"))

                await cl.Message(
                    author=assistant.name,
                    content=file_streams,
                ).send()

            elif file.mime in ["text/plain", "text/markdown"]:
                # Open text-based files in 'r' mode and encode them as UTF-8
                file_streams.append(file_path.open("r", encoding="utf-8"))
                    

        # Handle image files (binary)
        elif file.mime in ["image/png", "image/jpeg"]:
            await cl.Message(
                author=assistant.name,
                content="Uploading image file... with mime type: " + file.mime,
            ).send()

            # Open image files in 'rb' mode
            with file_path.open("rb") as f:
                image_files.append(f.read())


    if file_streams:
        file_batch = add_files_to_vector_store(file_streams)
        print(file_batch)

    """

    # Process files and get content and tools
    tools, content = await process_files(message.elements)


    print(f"Message content: {message.content}")



    # Add a message to the thread with combined content and attachments
    oai_message = await async_openai_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message.content,
        attachments=tools,
    )

    # Create and Stream a Run
    async with async_openai_client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant.id,
        event_handler=EventHandler(assistant_name=assistant.name),
    ) as stream:
        await stream.until_done()

       




@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        # This is required for whisper to recognize the file type
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    # Write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)


@cl.on_audio_end
async def on_audio_end(elements: list[Element]):
    # Get the audio buffer from the session
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)  # Move the file pointer to the beginning
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")

    input_audio_el = cl.Audio(
        mime=audio_mime_type, content=audio_file, name=audio_buffer.name
    )
    await cl.Message(
        author="You",
        type="user_message",
        content="",
        elements=[input_audio_el, *elements],
    ).send()

    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    transcription = await speech_to_text(whisper_input)

    msg = cl.Message(author="You", content=transcription, elements=elements)

    await main(message=msg)