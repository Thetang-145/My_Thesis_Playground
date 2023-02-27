import torch
from transformers import BertTokenizer
from transformers.adapters import BertAdapterModel, AutoAdapterModel

# print("Load pre-trained BERT tokenizer from Huggingface.")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print("An input sentence.")
sentence = "It's also, clearly, great fun."

print("Tokenize the input sentence and create a PyTorch input tensor.")
input_data = tokenizer(sentence, return_tensors="pt")

print("Load pre-trained BERT model from HuggingFace Hub.")
# The `BertAdapterModel` class is specifically designed for working with adapters.
# It can be used with different prediction heads.
model = BertAdapterModel.from_pretrained('bert-base-uncased')

print("load pre-trained task adapter from Adapter Hub")
# this method call will also load a pre-trained classification head for the adapter task
adapter_name = model.load_adapter('sst-2@ukp', config='pfeiffer')

print("activate the adapter we just loaded, so that it is used in every forward pass")
model.set_active_adapters(adapter_name)

print("predict output tensor")
outputs = model(**input_data)

print("retrieve the predicted class label")
predicted = torch.argmax(outputs[0]).item()
assert predicted == 1

print("save model")
model.save_pretrained('saved/model/')
print("save adapter")
model.save_adapter('saved/adapter/', 'sst-2')

print("load model")
model = AutoAdapterModel.from_pretrained('saved/model/')
model.load_adapter('saved/adapter/')

# deactivate all adapters
model.set_active_adapters(None)
# delete the added adapter
model.delete_adapter('sst-2')