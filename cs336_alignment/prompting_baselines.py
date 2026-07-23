from vllm_utils import VLLMServer

def prompting_baselines():
   vllm_server = VLLMServer("allenai/OLMo-2-0425-1B")
   vllm_server.start()

if __name__ == "__main__":
    prompting_baselines()
