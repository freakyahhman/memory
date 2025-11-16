from llama_cpp import Llama

system_prompt = "You are an helpful assistant, always follow my instructions. To answer the question step by step, you can provide your retrieve request to assist you by the following json format:\n"

system_prompt += '''{
    "ASR": Optional[str]. The abstract information that people in the video may discuss, or just the summary of the question, in two sentences. If you don't need this information, please return null.
    "DET": Optional[list]. (The output must include only physical entities, not abstract concepts, less than five entities) All the physical entities and their location related to the question you want to retrieve, not abstract concepts. If you no need for this information, please return null.
    "OCR": Optional[list]. (The output must be specified as null or a list containing detailed texts in video that may relevant to the answer of the question. (The information that you want to know more about.)
    }
    ## Example 1: 
    Question: How many blue balloons are over the long table in the middle of the room at the end of this video? A. 1. B. 2. C. 3. D. 4.
    Your retrieve can be:
    {
        "ASR": "The location and the color of balloons, the number of the blue balloons.",
        "DET": ["blue ballons", "long table"],
        "OCR": null
    }
    ## Example 2: 
    Question: In the lower left corner of the video, what color is the woman wearing on the right side of the man in black clothes? A. Blue. B. White. C. Red. D. Yellow.
    Your retrieve can be:
    {
        "ASR": null,
        "DET": ["the man in black", "woman"],
        "OCR": null
    }
    ## Example 3: 
    Question: In which country is the comedy featured in the video recognized worldwide? A. China. B. UK. C. Germany. D. United States.
    Your retrieve can be:
    {
        "ASR": "The country recognized worldwide for its comedy.",
        "DET": null,
        "OCR": ["China", "UK", "Germany", "USA"]
    }
    Note that you don't need to answer the question in this step, so you don't need any infomation about the video of image. You only need to provide your retrieve request (it's optional), and I will help you retrieve the infomation you want. Please provide the json format.'''


question = "According to the video, what does Sam said about China and Vietnam relationship."

process = "Question: " + question

llm = Llama(
        model_path="gemma-3-4b-it-qat-Q4_K_M.gguf",
        n_ctx=4096,
        n_threads=4,
        n_gpu_layers=-1
    )

output = llm.create_chat_completion(
      messages = [
          {   "role": "system",
              "content": system_prompt},
          {
              "role": "user",
              "content": process
          }
      ]
)

print(output["choices"][0]["message"]["content"])