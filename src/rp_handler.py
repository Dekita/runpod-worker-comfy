import runpod
from runpod.serverless.utils import rp_upload
import json
import urllib.request
import urllib.parse
import time
import os
import requests
import base64

# import nudenet lib: (nudity detection)
# see: https://github.com/notAI-tech/NudeNet/tree/v3
from nudenet import NudeDetector


# Time to wait between API check attempts in milliseconds
COMFY_API_AVAILABLE_INTERVAL_MS = 50
# Maximum number of API check attempts
COMFY_API_AVAILABLE_MAX_RETRIES = 500
# Time to wait between poll attempts in milliseconds
COMFY_POLLING_INTERVAL_MS = 250
# Maximum number of poll attempts
COMFY_POLLING_MAX_RETRIES = 999
# Host where ComfyUI is running
COMFY_HOST = "127.0.0.1:8188"
# The path where ComfyUI stores the generated images
COMFY_OUTPUT_PATH = "/comfyui/output"


def log(string):
    """
    Logs string to console with basic system prefix
    
    Args:
    - string (str): The string to log
    """
    print(f"[runpod-worker-comfy] {string}")


def job_prop_to_bool(job_input, propname):
    """
    Returns boolean based on variable value.
    Allows for checking string booleans, eg: "true"

    Args:
    - job_input (dict): A dictionary containing job input parameters.
    
    Returns: 
    bool: True if job_input dict has propname that seems bool-ish
    """
    value = job_input.get(propname)
    if value is None: return False
    if isinstance(value, bool): return value
    true_strings = ['true', 't', 'yes', 'y', 'ok', '1']
    if isinstance(value, str): return value.lower().strip() in true_strings
    return False


def return_error(error_message):
    """
    Logs error message and then returns basic dict with "error" prop using message

    Args:
    - error_message (string): A string containing the error emssage to show
    
    Returns: 
    dict: containing error property with error message
    """    
    log(error_message) # log message then return error value
    return {"error": error_message}


def check_server(url, retries=50, delay=500):
    """
    Check if a server is reachable via HTTP GET request

    Args:
    - url (str): The URL to check
    - retries (int, optional): The number of times to attempt connecting to the server. Default is 50
    - delay (int, optional): The time in milliseconds to wait between retries. Default is 500

    Returns:
    bool: True if the server is reachable within the given number of retries, otherwise False
    """
    for i in range(retries):
        try:
            response = requests.get(url)
            # If the response status code is 200, the server is up and running
            if response.status_code == 200:
                log(f"API is reachable")
                return True
        except requests.RequestException as e:
            # If an exception occurs, the server may not be ready
            pass

        # Wait for the specified delay before retrying
        time.sleep(delay/1000)

    log(f"Failed to connect to server at {url} after {retries} attempts.")
    return False


def queue_workflow(workflow):
    """
    Queue a workflow to be processed by ComfyUI

    Args:
        workflow (dict): A dictionary containing the workflow to be processed

    Returns:
        dict: The JSON response from ComfyUI after processing the prompt
    """
    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(f"http://{COMFY_HOST}/prompt", data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_history(prompt_id):
    """
    Retrieve the history of a given prompt using its ID

    Args:
        prompt_id (str): The ID of the prompt whose history is to be retrieved

    Returns:
        dict: The history of the prompt, containing all the processing steps and results
    """
    with urllib.request.urlopen(f"http://{COMFY_HOST}/history/{prompt_id}") as response:
        return json.loads(response.read())


def base64_encode(img_file):
    """
    Returns base64 encoded image.
    """
    log(f"scanning base64 for {img_file}")
    with open(img_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")
    

def detect_nudity(img_file):
    """
    Returns:
    list: of detected nudity for img_file if able
    """
    log(f"scanning nudity for {img_file}")
    return NudeDetector().detect(img_file) 


def handler(job):
    """
    The main function that handles a job of generating an image.

    This function validates the input, sends a prompt to ComfyUI for processing,
    polls ComfyUI for result, and retrieves generated images.

    Args:
        job (dict): A dictionary containing job details and input parameters.

    Returns:
        dict: A dictionary containing either an error message or a success status with generated images.
    """
    job_input = job["input"]
    job_output = {}

    # Validate inputs
    if job_input is None:
        return return_error(f"no 'input' property found on job data")

    if job_input.get("workflow") is None:
        return return_error(f"no 'workflow' property found on job data")
    
    workflow = job_input.get("workflow")

    # if workflow is a string then try convert to json
    if isinstance(workflow, str):
        try:
            workflow = json.loads(workflow)
        except json.JSONDecodeError:
            return return_error(f"Invalid JSON format in 'workflow' data")
        
    # ensure workflow is valid JSON:
    if not isinstance(workflow, dict):
        return return_error(f"'workflow' must be a JSON object or JSON-encoded string")
    
    # Make sure that the ComfyUI API is available
    check_server(
        f"http://{COMFY_HOST}",
        COMFY_API_AVAILABLE_MAX_RETRIES,
        COMFY_API_AVAILABLE_INTERVAL_MS,
    )

    # Queue the prompt
    try:
        queued = queue_workflow(workflow)
        comfy_job_id = queued["prompt_id"]
        log(f"ComfyUI queued with job ID {comfy_job_id}")
    except Exception as e:
        return return_error(f"Error queuing prompt: {str(e)}")

    # Poll for completion
    log(f"wait until image generation is complete")
    retries = 0
    try:
        while retries < COMFY_POLLING_MAX_RETRIES:
            history = get_history(comfy_job_id)

            # Exit the loop if we have found the history
            if comfy_job_id in history and history[comfy_job_id].get("outputs"):
                break
            else:
                # Wait before trying again
                time.sleep(COMFY_POLLING_INTERVAL_MS / 1000)
                retries += 1
        else:
            return return_error(f"Max retries reached while waiting for image generation")
    except Exception as e:
        return return_error(f"Error waiting for image generation: {str(e)}")

    # Fetching generated images
    output_images = {}

    outputs = history[comfy_job_id].get("outputs")

    for node_id, node_output in outputs.items():
        if "images" in node_output:
            for image in node_output["images"]:
                output_images = image["filename"]

    log(f"image generation is done")

    # expected image output folder
    local_image_path = f"{COMFY_OUTPUT_PATH}/{output_images}"
    # The image is in the output folder
    if os.path.exists(local_image_path):
        log("the image exists in the output folder")
        # use runpod upload to attempt aws image upload
        # will only work when aws credts have been set in env, 
        # ~ or are given when sending the job request
        image_url = rp_upload.upload_image(job["id"], local_image_path)
        # check image_url to see if was uploaded to aws
        aws_uploaded = "simulated_uploaded/" not in image_url
        # setup base return object structure
        job_output["url"] = image_url
        # check generated image for nudity if flag set
        if job_prop_to_bool(job_input, "return-nsfw"):
            job_output["nsfw"] = detect_nudity(local_image_path)
        # convert generated image to base64 if not uploaded to aws and able
        if not aws_uploaded and job_prop_to_bool(job_input, "return-b64"):
            job_output["base64"] = base64_encode(local_image_path)
        # else return image url
        return job_output
    # image wasnt found in the output folder, no need for else
    return return_error(f"image does not exist in the specified output folder: {local_image_path}")


# Start the handler
runpod.serverless.start({"handler": handler})
