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



AZURE_OPENAI_API_VERSION = os.environ.get('AZURE_OPENAI_API_VERSION')
AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT')

async_openai_client = AsyncAzureOpenAI(api_key=AZURE_OPENAI_API_KEY, api_version=AZURE_OPENAI_API_VERSION, azure_endpoint=AZURE_OPENAI_ENDPOINT)
sync_openai_client = client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,  
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint = AZURE_OPENAI_ENDPOINT
    )

available_functions = {"get_weather": get_weather}
verbose_output = True

assistant = client.beta.assistants.create(
    model="gpt-4o-2024-05-13", # replace with model deployment name.
    instructions="You are a helpful product support assistant and you answer questions based on the files provided to you.",
    tools=[{"type":"file_search"},{"type":"code_interpreter"},{"type":"function","function":{"name":"get_weather","description":"Determine weather in my location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state e.g. Seattle, WA"},"unit":{"type":"string","enum":["c","f"]}},"required":["location"]}}}],
    tool_resources={"file_search":{"vector_store_ids":["vs_enHy9WWIxQXbgZM5WIcdTyCd"]},"code_interpreter":{"file_ids":[]}},
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
                        print(f"Error: {output.logs}")
                        error_step.language = "markdown"
                        error_step.start = self.current_step.start
                        error_step.end = utc_now()
                        await error_step.send()
            else:
                if delta.code_interpreter.input:
                    print(f"Code Interpreter Input: {delta.code_interpreter.input}")
                    await self.current_step.stream_token(delta.code_interpreter.input)
        elif delta.type == "file_search":
            print(f"File Search Delta: {delta.file_search}")
        
            

    async def on_event(self, event):
        thread_id = cl.user_session.get("thread_id")

        print(f"Event: {event.event} \n -------------------------------------------------- \n")

        if event.event == 'thread.run.requires_action':

            print(f"Event: {event.data} \n -------------------------------------------------- \n")
            if event.data.status == "requires_action":
                tool_responses = []
                if (
                    event.data.required_action.type == "submit_tool_outputs"
                    and event.data.required_action.submit_tool_outputs.tool_calls is not None
                ):
                    tool_calls = event.data.required_action.submit_tool_outputs.tool_calls

                    print(f"Tool Calls: {len(tool_calls)} \n -------------------------------------------------- \n")

                    self.current_step = cl.Step(name="functions", type="tool") 
                    self.current_step.created_at = utc_now()       
                   


                    for call in tool_calls:
                        if call.type == "function":
                            if call.function.name not in available_functions:
                                raise Exception("Function requested by the model does not exist")
                            function_to_call = available_functions[call.function.name]
                            tool_response = function_to_call(json.loads(call.function.arguments)["location"])
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
                    


        if event.event == 'thread.run.completed':
            print(f"Event done: {event.data} \n -------------------------------------------------- \n")


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

    upload_files = client.beta.vector_stores.files.upload_and_poll(files)
    file_ids = []
    for file in files:
        uploaded_file = await async_openai_client.files.create(
            file=Path(file.path), purpose="assistants"
        )
        file_ids.append(uploaded_file.id)
    return file_ids


async def process_files(files: List[Element]):
    # Upload files if any and get file_ids
    file_ids = []
    if len(files) > 0:
        file_ids = await upload_files(files)

    return [
        {
            "file_id": file_id,
            "tools": [{"type": "code_interpreter"}, {"type": "file_search"}] if file.mime in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/markdown", "application/pdf", "text/plain"] else [{"type": "code_interpreter"}],
        }
        for file_id, file in zip(file_ids, files)
    ]


@cl.on_chat_start
async def start_chat():
    # Create a Thread
    thread = await async_openai_client.beta.threads.create()
    # Store thread ID in user session for later use
    cl.user_session.set("thread_id", thread.id)
    await cl.Avatar(name=assistant.name, path="./public/logo.png").send()
    await cl.Message(content=f"Hello, I'm {assistant.name}!", disable_feedback=True).send()
    

@cl.on_message
async def main(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")

    attachments = await process_files(message.elements)

    # Add a Message to the Thread
    oai_message = await async_openai_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message.content,
        attachments=attachments,
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