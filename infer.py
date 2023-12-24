from util.model import get_peft, load_roberta, predict
from peft.peft_model import PeftModel

roberta = load_roberta()
peft = PeftModel.from_pretrained(
    roberta.model, "./model/roberta-text-classification-lora"
)
print("Model loaded.")

label_name = {0: "0 Negative", 1: "1 Positive"}
while True:
    input_sentence = input("Input the sentence ( or enter to exit ): ")
    if input_sentence == "":
        break
    output = predict(peft, roberta.tokenizer, input_sentence)
    print(f"It is {label_name[output]}.")
