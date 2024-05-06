# Align is not Enough: Multimodal Universal Jailbreak Attack against Multimodal Large Language Models.
This is the official implementation for the multimodal universal jailbreak attack.

## Dependencies
For MiniGPT4:
```
pip install -r ./MiniGPT-v2/requirement_minigpt4.txt
```

For LLaVA:
```
pip install -r ./LLaVA/requirement_llava.txt
```

## Usage
- **Dataset**

    -   The dataset in the experiments is from the GCG https://github.com/llm-attacks/llm-attacks

- **Victim Models**
    - MiniGPT-v2-7B
        - checkpoints: https://drive.google.com/file/d/1aVbfW7nkCSYx99_vCRyP1sOlQiWVSnAl/view
    - MiniGPT4-7B
        - checkpoints: https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view
    - LLaVA
        - llava-v1.5-7B: https://huggingface.co/liuhaotian/llava-v1.5-7b
        - llava-v1.5-13B: https://huggingface.co/liuhaotian/llava-v1.5-13b
        - llava-v1.6-34B: https://huggingface.co/llava-hf/llava-v1.6-34b-hf
    - Yi
        - Yi-VL-34B: https://huggingface.co/01-ai/Yi-VL-34B
    - InstructBLIP
        - InstructBLUP-vicuna-7B: https://huggingface.co/Salesforce/instructblip-vicuna-7b
        - InstructBLUP-vicuna-13B: https://huggingface.co/Salesforce/instructblip-vicuna-13b

- **Run**
For the multimodal universal jailbreak attack on MiniGPT-v2 in a white-box setting:
```
python ./MiniGPT-v2/jailbreak_attack/main.py
```

For the multimodal universal jailbreak attack on LLaVA in a white-box setting:
```
python ./llava/jailbreak_attack/main_llava.py
```